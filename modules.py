import utils

import numpy as np

import torch
from torch import nn
from typing import Tuple, Optional, Dict, Any, Union


class ContrastiveSWM(nn.Module):
    """Main module for a Contrastively-trained Structured World Model (C-SWM).

    Args:
        embedding_dim: Dimensionality of abstract state space.
        input_dims: Shape of input observation.
        hidden_dim: Number of hidden units in encoder and transition model.
        action_dim: Dimensionality of action space.
        num_objects: Number of object slots.
    """
    def __init__(self, embedding_dim: int, input_dims: Tuple[int, ...], hidden_dim: int, action_dim: int,
                 num_objects: int, hinge: float = 1., sigma: float = 0.5, encoder: str = 'large',
                 ignore_action: bool = False, copy_action: bool = False):
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.embedding_dim: int = embedding_dim
        self.action_dim: int = action_dim
        self.num_objects: int = num_objects
        self.hinge: float = hinge
        self.sigma: float = sigma
        self.ignore_action: bool = ignore_action
        self.copy_action: bool = copy_action
        
        self.pos_loss: float = 0
        self.neg_loss: float = 0

        num_channels: int = input_dims[0]
        width_height: Tuple[int, ...] = input_dims[1:]

        if encoder == 'small':
            self.obj_extractor: nn.Module = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 10
        elif encoder == 'medium':
            self.obj_extractor: nn.Module = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)
            # CNN image size changes
            width_height = np.array(width_height)
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor: nn.Module = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=hidden_dim // 16,
                num_objects=num_objects)

        self.obj_encoder: nn.Module = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_objects=num_objects)

        self.transition_model: nn.Module = TransitionGNN(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_objects=num_objects,
            ignore_action=ignore_action,
            copy_action=copy_action)

        self.width: int = width_height[0]
        self.height: int = width_height[1]

    def energy(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, no_trans: bool = False) -> torch.Tensor:
        """Energy function based on normalized squared L2 norm."""
        print(f"Energy inputs - state: {state.shape}, action: {action.shape}, next_state: {next_state.shape}")

        norm: float = 0.5 / (self.sigma**2)

        if no_trans:
            diff: torch.Tensor = state - next_state
        else:
            pred_trans: torch.Tensor = self.transition_model(state, action)
            diff: torch.Tensor = state + pred_trans - next_state

        return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:

        objs: torch.Tensor = self.obj_extractor(obs)
        next_objs: torch.Tensor = self.obj_extractor(next_obs)

        state: torch.Tensor = self.obj_encoder(objs)
        next_state: torch.Tensor = self.obj_encoder(next_objs)

        # # Sample negative state across episodes at random
        # batch_size: int = state.size(0)
        # perm: np.ndarray = np.random.permutation(batch_size)
        # neg_state: torch.Tensor = state[perm]

        # Sample multiple negative states across episodes at random
        batch_size: int = state.size(0)
        num_negatives: int = 10  # Number of negative samples
        neg_states: torch.Tensor = torch.stack([state[np.random.permutation(batch_size)] for _ in range(num_negatives)], dim=1)

        print(f"state shape: {state.shape}")
        print(f"action shape: {action.shape}")
        print(f"next_state shape: {next_state.shape}")
        print(f"neg_states shape: {neg_states.shape}")

        self.pos_loss: float = self.energy(state, action, next_state)
        zeros: torch.Tensor = torch.zeros_like(self.pos_loss)
        
        pos_loss_per_example = self.pos_loss

        # negative loss using infoNCE
        expanded_state = state.unsqueeze(1).expand(-1, num_negatives, *state.shape[1:])  # Expand while preserving other dimensions
        expanded_action = action.unsqueeze(1).expand(-1, num_negatives, *action.shape[1:])  # Expand while preserving other dimensions
        neg_energy: torch.Tensor = self.energy(expanded_state, expanded_action, neg_states, no_trans=True)
        neg_energy: torch.Tensor = neg_energy.mean(dim=1)  # Average over negative samples

        # InfoNCE loss
        logits: torch.Tensor = torch.cat([pos_loss_per_example.unsqueeze(1), neg_energy], dim=1)
        labels: torch.Tensor = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        self.neg_loss: float = F.cross_entropy(logits, labels)

        # self.neg_loss: float = torch.max(
        #     zeros, self.hinge - self.energy(
        #         state, action, neg_state, no_trans=True)).mean()

        self.pos_loss = pos_loss_per_example.mean()
        loss: torch.Tensor = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obj_encoder(self.obj_extractor(obs))


class TransitionGNN(nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_objects: int,
                 ignore_action: bool = False, copy_action: bool = False, act_fn: str = 'relu'):
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.num_objects: int = num_objects
        self.ignore_action: bool = ignore_action
        self.copy_action: bool = copy_action

        if self.ignore_action:
            self.action_dim: int = 0
        else:
            self.action_dim: int = action_dim

        self.edge_mlp: nn.Module = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim))

        node_input_dim: int = hidden_dim + input_dim + self.action_dim

        self.node_mlp: nn.Module = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            utils.get_act_fn(act_fn),
            nn.Linear(hidden_dim, input_dim))

        self.edge_list: Optional[torch.Tensor] = None
        self.batch_size: int = 0

    def _edge_model(self, source: torch.Tensor, target: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        del edge_attr  # Unused.
        out: torch.Tensor = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is not None:
            row, col = edge_index
            agg: torch.Tensor = utils.unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out: torch.Tensor = torch.cat([node_attr, agg], dim=1)
        else:
            out: torch.Tensor = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size: int, num_objects: int, cuda: bool) -> torch.Tensor:
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            
            # Calculate offset for each batch
            offset = torch.arange(0, batch_size * num_objects, num_objects)
            offset = offset.view(-1, 1).expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1, 1)
            
            self.edge_list = self.edge_list + offset

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            # Convert to correct device
            if cuda:
                device = torch.cuda.current_device()
                self.edge_list = self.edge_list.cuda(device)

        return self.edge_list

    def forward(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        cuda: bool = states.is_cuda
        batch_size: int = states.size(0)
        num_nodes: int = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr: torch.Tensor = states.view(-1, self.input_dim)

        edge_attr: Optional[torch.Tensor] = None
        edge_index: Optional[torch.Tensor] = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index: torch.Tensor = self._get_edge_list_fully_connected(
                batch_size, num_nodes, cuda)

            # Ensure edge_index is on the same device as node_attr
            if cuda and edge_index.device != node_attr.device:
                edge_index = edge_index.to(node_attr.device)

            row, col = edge_index
            edge_attr: torch.Tensor = self._edge_model(
                node_attr[row], node_attr[col], edge_attr)

        if not self.ignore_action:

            if self.copy_action:
                action_vec: torch.Tensor = utils.to_one_hot(
                    action, self.action_dim).repeat(1, self.num_objects)
                action_vec: torch.Tensor = action_vec.view(-1, self.action_dim)
            else:
                action_vec: torch.Tensor = utils.to_one_hot(
                    action, self.action_dim * num_nodes)
                action_vec: torch.Tensor = action_vec.view(-1, self.action_dim)

            # Attach action to each state
            node_attr: torch.Tensor = torch.cat([node_attr, action_vec], dim=-1)

        node_attr: torch.Tensor = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class EncoderCNNSmall(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, act_fn: str = 'sigmoid',
                 act_fn_hid: str = 'relu'):
        super().__init__()
        self.cnn1: nn.Module = nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2: nn.Module = nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1: nn.Module = nn.BatchNorm2d(hidden_dim)
        self.act1: nn.Module = utils.get_act_fn(act_fn_hid)
        self.act2: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))
    
    
class EncoderCNNMedium(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, act_fn: str = 'sigmoid',
                 act_fn_hid: str = 'leaky_relu'):
        super().__init__()

        self.cnn1: nn.Module = nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1: nn.Module = utils.get_act_fn(act_fn_hid)
        self.ln1: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.cnn2: nn.Module = nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.ln1(self.cnn1(obs)))
        h: torch.Tensor = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, act_fn: str = 'sigmoid',
                 act_fn_hid: str = 'relu'):
        super().__init__()

        self.cnn1: nn.Module = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1: nn.Module = utils.get_act_fn(act_fn_hid)
        self.ln1: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.cnn2: nn.Module = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2: nn.Module = utils.get_act_fn(act_fn_hid)
        self.ln2: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.cnn3: nn.Module = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3: nn.Module = utils.get_act_fn(act_fn_hid)
        self.ln3: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.cnn4: nn.Module = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.ln1(self.cnn1(obs)))
        h: torch.Tensor = self.act2(self.ln2(self.cnn2(h)))
        h: torch.Tensor = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(nn.Module):
    """MLP encoder, maps observation to latent state."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_objects: int,
                 act_fn: str = 'relu'):
        super().__init__()

        self.num_objects: int = num_objects
        self.input_dim: int = input_dim

        self.fc1: nn.Module = nn.Linear(self.input_dim, hidden_dim)
        self.fc2: nn.Module = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Module = nn.Linear(hidden_dim, output_dim)

        self.ln: nn.Module = nn.LayerNorm(hidden_dim)

        self.act1: nn.Module = utils.get_act_fn(act_fn)
        self.act2: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        h_flat: torch.Tensor = ins.view(-1, self.num_objects, self.input_dim)
        h: torch.Tensor = self.act1(self.fc1(h_flat))
        h: torch.Tensor = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(nn.Module):
    """MLP decoder, maps latent state to image."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, output_size: Tuple[int, ...],
                 act_fn: str = 'relu'):
        super().__init__()

        self.fc1: nn.Module = nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2: nn.Module = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Module = nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim: int = input_dim
        self.num_objects: int = num_objects
        self.output_size: Tuple[int, ...] = output_size

        self.act1: nn.Module = utils.get_act_fn(act_fn)
        self.act2: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        obj_ids: torch.Tensor = torch.arange(self.num_objects)
        obj_ids: torch.Tensor = utils.to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids: torch.Tensor = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h: torch.Tensor = torch.cat((ins, obj_ids), -1)
        h: torch.Tensor = self.act1(self.fc1(h))
        h: torch.Tensor = self.act2(self.fc2(h))
        h: torch.Tensor = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, output_size: Tuple[int, ...],
                 act_fn: str = 'relu'):
        super().__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim: int = width * height

        self.fc1: nn.Module = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Module = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Module = nn.Linear(hidden_dim, output_dim)
        self.ln: nn.Module = nn.LayerNorm(hidden_dim)

        self.deconv1: nn.Module = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2: nn.Module = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim: int = input_dim
        self.num_objects: int = num_objects
        self.map_size: Tuple[int, int, int] = output_size[0], width, height

        self.act1: nn.Module = utils.get_act_fn(act_fn)
        self.act2: nn.Module = utils.get_act_fn(act_fn)
        self.act3: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.fc1(ins))
        h: torch.Tensor = self.act2(self.ln(self.fc2(h)))
        h: torch.Tensor = self.fc3(h)

        h_conv: torch.Tensor = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h: torch.Tensor = self.act3(self.deconv1(h_conv))
        return self.deconv2(h)


class DecoderCNNMedium(nn.Module):
    """CNN decoder, maps latent state to image."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, output_size: Tuple[int, ...],
                 act_fn: str = 'relu'):
        super().__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim: int = width * height

        self.fc1: nn.Module = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Module = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Module = nn.Linear(hidden_dim, output_dim)
        self.ln: nn.Module = nn.LayerNorm(hidden_dim)

        self.deconv1: nn.Module = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2: nn.Module = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.input_dim: int = input_dim
        self.num_objects: int = num_objects
        self.map_size: Tuple[int, int, int] = output_size[0], width, height

        self.act1: nn.Module = utils.get_act_fn(act_fn)
        self.act2: nn.Module = utils.get_act_fn(act_fn)
        self.act3: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.fc1(ins))
        h: torch.Tensor = self.act2(self.ln(self.fc2(h)))
        h: torch.Tensor = self.fc3(h)

        h_conv: torch.Tensor = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h: torch.Tensor = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim: int, hidden_dim: int, num_objects: int, output_size: Tuple[int, ...],
                 act_fn: str = 'relu'):
        super().__init__()

        width, height = output_size[1], output_size[2]

        output_dim: int = width * height

        self.fc1: nn.Module = nn.Linear(input_dim, hidden_dim)
        self.fc2: nn.Module = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Module = nn.Linear(hidden_dim, output_dim)
        self.ln: nn.Module = nn.LayerNorm(hidden_dim)

        self.deconv1: nn.Module = nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2: nn.Module = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3: nn.Module = nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4: nn.Module = nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1: nn.Module = nn.BatchNorm2d(hidden_dim)
        self.ln2: nn.Module = nn.BatchNorm2d(hidden_dim)
        self.ln3: nn.Module = nn.BatchNorm2d(hidden_dim)

        self.input_dim: int = input_dim
        self.num_objects: int = num_objects
        self.map_size: Tuple[int, int, int] = output_size[0], width, height

        self.act1: nn.Module = utils.get_act_fn(act_fn)
        self.act2: nn.Module = utils.get_act_fn(act_fn)
        self.act3: nn.Module = utils.get_act_fn(act_fn)
        self.act4: nn.Module = utils.get_act_fn(act_fn)
        self.act5: nn.Module = utils.get_act_fn(act_fn)

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.act1(self.fc1(ins))
        h: torch.Tensor = self.act2(self.ln(self.fc2(h)))
        h: torch.Tensor = self.fc3(h)

        h_conv: torch.Tensor = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h: torch.Tensor = self.act3(self.ln1(self.deconv1(h_conv)))
        h: torch.Tensor = self.act4(self.ln1(self.deconv2(h)))
        h: torch.Tensor = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)
