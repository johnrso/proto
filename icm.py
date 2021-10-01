"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(IntrinsicCuriosityModule, self).__init__()
        self.conv = BaseConv(num_inputs)
        self.feature_size = 64 * 6 * 6
        self.inverse_net = nn.Sequential(
            nn.Linear(self.feature_size * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, num_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(self.feature_size + num_actions, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.feature_size)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state, next_state, action):
        state_ft = self.conv(state)
        next_state_ft = self.conv(next_state)
        state_ft = state_ft.view(-1, self.feature_size)
        next_state_ft = next_state_ft.view(-1, self.feature_size)
        return self.inverse_net(torch.cat((state_ft, next_state_ft), 1)), self.forward_net(
            torch.cat((state_ft, action), 1)), next_state_ft
