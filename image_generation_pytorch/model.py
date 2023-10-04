from typing import Literal
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from numpy import sqrt
import torch
from torch.nn.functional import linear


class Generator(nn.Module):
    def __init__(
        self,
        g_layers: list[int] = [],
        activation_fn: bool = True,
        inject_z: bool = True,
        transform_rep: int = 6,
        transform_z: bool = False,
        concat_injection: bool = False,
        norm: Literal["instance", "batch"] = "instance",
    ):
        super(Generator, self).__init__()
        self.activation_fn = activation_fn
        self.g_layers = g_layers
        self.inject_z = inject_z
        self.num_layers = len(self.g_layers) - 1  # minus the input/output sizes
        self.transform_rep = transform_rep
        self.transform_z = transform_z
        self.concat_injection = concat_injection
        self.norm = norm

        if self.transform_z:
            for i in range(self.transform_rep):
                setattr(
                    self,
                    f"global{i}",
                    nn.Sequential(
                        nn.Linear(self.g_layers[0], self.g_layers[0]), nn.ReLU(inplace=False)
                    ),
                )

        for i in range(self.num_layers):
            if i == 0:
                if not self.activation_fn:
                    if self.norm == "instance":
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    1,
                                    0,
                                    bias=False,
                                ),
                                nn.InstanceNorm2d(self.g_layers[i + 1]),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    1,
                                    0,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.g_layers[i + 1]),
                            ),
                        )
                else:
                    if self.norm == "instance":
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    1,
                                    0,
                                    bias=False,
                                ),
                                nn.InstanceNorm2d(self.g_layers[i + 1]),
                                nn.ReLU(inplace=False),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    1,
                                    0,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.g_layers[i + 1]),
                                nn.ReLU(inplace=False),
                            ),
                        )

            elif i == self.num_layers - 1:
                if not self.activation_fn:
                    setattr(
                        self,
                        f"layer{i}",
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                self.g_layers[i],
                                self.g_layers[i + 1],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.Tanh(),
                        ),
                    )
                else:
                    setattr(
                        self,
                        "layer{}".format(i),
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                self.g_layers[i],
                                self.g_layers[i + 1],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.Tanh(),
                        ),
                    )
            else:
                if self.inject_z:
                    if not self.activation_fn:
                        setattr(
                            self,
                            "inject{}".format(i),
                            nn.Linear(self.g_layers[0], self.g_layers[i]),
                        )
                    else:
                        setattr(
                            self,
                            "inject{}".format(i),
                            nn.Sequential(
                                nn.Linear(self.g_layers[0], self.g_layers[i]), nn.ReLU(inplace=False)
                            ),
                        )
                if not self.activation_fn:
                    if self.norm == "instance":
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.InstanceNorm2d(self.g_layers[i + 1]),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.g_layers[i + 1]),
                            ),
                        )
                    if self.concat_injection:
                        if self.norm == "instance":
                            setattr(
                                self,
                                "layer{}".format(i),
                                nn.Sequential(
                                    nn.ConvTranspose2d(
                                        2 * self.g_layers[i],
                                        self.g_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.InstanceNorm2d(self.g_layers[i + 1]),
                                ),
                            )
                        else:
                            setattr(
                                self,
                                "layer{}".format(i),
                                nn.Sequential(
                                    nn.ConvTranspose2d(
                                        2 * self.g_layers[i],
                                        self.g_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.g_layers[i + 1]),
                                ),
                            )
                else:
                    if self.norm == "instance":
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.InstanceNorm2d(self.g_layers[i + 1]),
                                nn.ReLU(inplace=False),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.ConvTranspose2d(
                                    self.g_layers[i],
                                    self.g_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.g_layers[i + 1]),
                                nn.ReLU(inplace=False),
                            ),
                        )
                    if self.concat_injection:
                        if self.norm == "instance":
                            setattr(
                                self,
                                "layer{}".format(i),
                                nn.Sequential(
                                    nn.ConvTranspose2d(
                                        2 * self.g_layers[i],
                                        self.g_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.InstanceNorm2d(self.g_layers[i + 1]),
                                    nn.ReLU(inplace=False),
                                ),
                            )
                        else:
                            setattr(
                                self,
                                "layer{}".format(i),
                                nn.Sequential(
                                    nn.ConvTranspose2d(
                                        2 * self.g_layers[i],
                                        self.g_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(self.g_layers[i + 1]),
                                    nn.ReLU(inplace=False),
                                ),
                            )

    def forward(self, x):
        z = x.squeeze(3).squeeze(2)
        if self.transform_z:
            for i in range(self.transform_rep):
                z = getattr(self, "global{}".format(i))(z)
        x = z.unsqueeze(2).unsqueeze(3)
        for i in range(self.num_layers):
            if self.inject_z:
                if i > 0 and i < self.num_layers - 1:
                    a = getattr(self, "inject{}".format(i))(z)
                    a = (
                        a.unsqueeze(2)
                        .unsqueeze(3)
                        .expand(x.size(0), x.size(1), x.size(2), x.size(3))
                    )
                    if self.concat_injection:
                        x = torch.cat((x, a), dim=1)
                    else:
                        x *= a
            x = getattr(self, "layer{}".format(i))(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        d_layers: list[int] = [],
        activation_fn: bool = True,
        spectral_norm: bool = False,
    ):
        super(Discriminator, self).__init__()
        self.activation_fn = activation_fn
        self.d_layers = d_layers
        self.num_layers = len(self.d_layers) - 1  # minus the input/output sizes
        for i in range(self.num_layers):
            if i == 0:
                if not spectral_norm:
                    if not self.activation_fn:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Conv2d(
                                self.d_layers[i],
                                self.d_layers[i + 1],
                                4,
                                2,
                                1,
                                bias=False,
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.Conv2d(
                                    self.d_layers[i],
                                    self.d_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.LeakyReLU(0.2, inplace=False),
                            ),
                        )
                else:
                    if not self.activation_fn:
                        setattr(
                            self,
                            "layer{}".format(i),
                            spectral_norm(
                                nn.Conv2d(
                                    self.d_layers[i],
                                    self.d_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                )
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                spectral_norm(
                                    nn.Conv2d(
                                        self.d_layers[i],
                                        self.d_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    )
                                ),
                                nn.LeakyReLU(0.2, inplace=False),
                            ),
                        )
            elif i == self.num_layers - 1:
                if not spectral_norm:
                    setattr(
                        self,
                        "layer{}".format(i),
                        nn.Conv2d(
                            self.d_layers[i], self.d_layers[i + 1], 4, 2, 1, bias=False
                        ),
                    )
                else:
                    setattr(
                        self,
                        "layer{}".format(i),
                        spectral_norm(
                            nn.Conv2d(
                                self.d_layers[i],
                                self.d_layers[i + 1],
                                4,
                                2,
                                1,
                                bias=False,
                            )
                        ),
                    )
            else:
                if not spectral_norm:
                    if not self.activation_fn:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.Conv2d(
                                    self.d_layers[i],
                                    self.d_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.d_layers[i + 1]),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.Conv2d(
                                    self.d_layers[i],
                                    self.d_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.d_layers[i + 1]),
                                nn.LeakyReLU(0.2, inplace=False),
                            ),
                        )
                else:
                    if not self.activation_fn:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                spectral_norm(
                                    nn.Conv2d(
                                        self.d_layers[i],
                                        self.d_layers[i + 1],
                                        4,
                                        2,
                                        1,
                                        bias=False,
                                    )
                                ),
                                nn.BatchNorm2d(self.d_layers[i + 1]),
                            ),
                        )
                    else:
                        setattr(
                            self,
                            "layer{}".format(i),
                            nn.Sequential(
                                nn.Conv2d(
                                    self.d_layers[i],
                                    self.d_layers[i + 1],
                                    4,
                                    2,
                                    1,
                                    bias=False,
                                ),
                                nn.BatchNorm2d(self.d_layers[i + 1]),
                                nn.LeakyReLU(0.2, inplace=False),
                            ),
                        )

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"layer{i}")(x)
        return x
