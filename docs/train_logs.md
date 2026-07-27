```bash
wandb: WARNING Saving files without folders. If you want to preserve subdirectories pass base_path to wandb.save, i.e. wandb.save("/mnt/folder/file.h5", base_path="/mnt")
invalid syntax (<string>, line 1)
[2026-07-21 12:16:34,916][torch_points3d.applications.minkowski][WARNING] - Minkowski API is deprecated in favor of the SparseConv3d API. It should be a simple drop in replacement (no change to the API).
[2026-07-21 12:16:34,918][torch_points3d.applications.modelfactory][INFO] - The config will be used to build the model
[2026-07-21 12:16:35,159][torch_points3d.applications.minkowski][WARNING] - Minkowski API is deprecated in favor of the SparseConv3d API. It should be a simple drop in replacement (no change to the API).
[2026-07-21 12:16:35,161][torch_points3d.applications.modelfactory][INFO] - The config will be used to build the model
[2026-07-21 12:16:35,272][torch_points3d.applications.minkowski][WARNING] - Minkowski API is deprecated in favor of the SparseConv3d API. It should be a simple drop in replacement (no change to the API).
[2026-07-21 12:16:35,273][torch_points3d.applications.modelfactory][INFO] - The config will be used to build the model
[2026-07-21 12:16:35,399][torch_points3d.core.schedulers.bn_schedulers][INFO] - Setting batchnorm momentum at 0.1
[2026-07-21 12:16:35,411][torch_points3d.models.base_model][WARNING] - The path does not exist, it will not load any model
[2026-07-21 12:16:35,412][torch_points3d.trainer][INFO] - PointGroup3heads(
  (Backbone): MinkowskiUnet(
    (down_modules): ModuleList(
      (0): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=4, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (1): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (2): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=32, out=48, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (3): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=48, out=48, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=48, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=48, out=64, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (4): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=64, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=64, out=80, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (5): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=80, out=80, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=80, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=80, out=96, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (6): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=96, out=96, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=96, out=112, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=112, out=112, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=96, out=112, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=112, out=112, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=112, out=112, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
    )
    (inner_modules): ModuleList(
      (0): Identity()
    )
    (up_modules): ModuleList(
      (0): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=112, out=112, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=112, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=112, out=96, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=96, out=96, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (1): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=192, out=192, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=192, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=192, out=80, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=80, out=80, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (2): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=160, out=160, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=160, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=160, out=64, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (3): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=128, out=128, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=128, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=128, out=48, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=48, out=48, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (4): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=96, out=96, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=96, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=96, out=32, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (5): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=16, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (6): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=32, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
    )
  )
  (ScorerUnet): MinkowskiUnet(
    (down_modules): ModuleList(
      (0): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (1): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=32, out=64, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
    )
    (inner_modules): ModuleList(
      (0): Identity()
    )
    (up_modules): ModuleList(
      (0): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=32, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (1): ResNetUp(
        (conv_in): Seq(
          (0): MinkowskiConvolutionTranspose(in=64, out=64, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolutionTranspose(in=64, out=16, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolutionTranspose(in=16, out=16, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
    )
  )
  (ScorerEncoder): MinkowskiEncoder(
    (down_modules): ModuleList(
      (0): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=16, out=16, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=16, out=32, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
      (1): ResNetDown(
        (conv_in): Seq(
          (0): MinkowskiConvolution(in=32, out=32, kernel_size=[3, 3, 3], stride=[2, 2, 2], dilation=[1, 1, 1])
          (1): MinkowskiBatchNorm(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): MinkowskiReLU()
        )
        (blocks): Seq(
          (0): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=32, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
            (downsample): Seq(
              (0): MinkowskiConvolution(in=32, out=64, kernel_size=[1, 1, 1], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): ResBlock(
            (block): Seq(
              (0): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (1): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): MinkowskiReLU()
              (3): MinkowskiConvolution(in=64, out=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1])
              (4): MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (5): MinkowskiReLU()
            )
          )
        )
      )
    )
    (inner_modules): ModuleList(
      (0): GlobalBaseModule(
        (nn): Sequential(
          (0): Sequential(
            (0): Linear(in_features=64, out_features=16, bias=True)
            (1): FastBatchNorm1d(
              (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): LeakyReLU(negative_slope=0.2)
          )
        )
      )
    )
    (up_modules): ModuleList()
  )
  (ScorerMLP): Sequential(
    (0): Sequential(
      (0): Linear(in_features=16, out_features=16, bias=True)
      (1): FastBatchNorm1d(
        (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): LeakyReLU(negative_slope=0.2)
    )
    (1): Sequential(
      (0): Linear(in_features=16, out_features=16, bias=True)
      (1): FastBatchNorm1d(
        (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): LeakyReLU(negative_slope=0.2)
    )
  )
  (ScorerHead): Seq(
    (0): Linear(in_features=16, out_features=1, bias=True)
    (1): Sigmoid()
  )
  (Offset): Seq(
    (0): Sequential(
      (0): Sequential(
        (0): Linear(in_features=16, out_features=16, bias=False)
        (1): FastBatchNorm1d(
          (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): Linear(in_features=16, out_features=3, bias=True)
  )
  (Embed): Seq(
    (0): Sequential(
      (0): Sequential(
        (0): Linear(in_features=16, out_features=16, bias=False)
        (1): FastBatchNorm1d(
          (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): Linear(in_features=16, out_features=5, bias=True)
  )
  (Semantic): Seq(
    (0): Sequential(
      (0): Sequential(
        (0): Linear(in_features=16, out_features=16, bias=False)
        (1): FastBatchNorm1d(
          (batch_norm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): Linear(in_features=16, out_features=5, bias=True)
  )
  (LogSoftLay): Seq(
    (0): LogSoftmax(dim=-1)
  )
)
[2026-07-21 12:16:35,428][torch_points3d.utils.colors][INFO] - Optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    initial_lr: 0.001
    lr: 0.001
    weight_decay: 0
)
[2026-07-21 12:16:35,430][torch_points3d.utils.colors][INFO] - Learning Rate Scheduler: ExponentialLR({'gamma': 0.9885}, update_scheduler_on=on_epoch)
[2026-07-21 12:16:35,432][torch_points3d.utils.colors][INFO] - BatchNorm Scheduler: BNMomentumScheduler(base_momentum: 0.1, update_scheduler_on=on_epoch)
[2026-07-21 12:16:35,434][torch_points3d.utils.colors][INFO] - Accumulated gradients: None
[2026-07-21 12:16:35,453][torch_points3d.trainer][INFO] - Model size = 11872126
[2026-07-21 12:16:35,456][torch_points3d.trainer][INFO] - Dataset: TreeinsFusedDataset 
train_pre_batch_collate_transform = None
val_pre_batch_collate_transform = None
test_pre_batch_collate_transform = None
pre_transform = None
test_transform = Compose([
    XYZRelaFeature(axis=['x_rela', 'y_rela', 'z_rela']),
    XYZFeature(axis=['z']),
    AddFeatsByKeys(pos_x_rela=True, pos_y_rela=True, pos_z_rela=True, pos_z=True),
    Center(),
    GridSampling3D(grid_size=0.2, quantize_coords=True, mode=last),
])
train_transform = Compose([
    RandomNoise(sigma=0.01, clip=0.05),
    RandomRotate((-180, 180), axis=2),
    RandomScaleAnisotropic([0.9, 1.1]),
    Random symmetry of axes: x=True, y=False, z=False,
    XYZRelaFeature(axis=['x_rela', 'y_rela', 'z_rela']),
    XYZFeature(axis=['z']),
    AddFeatsByKeys(pos_x_rela=True, pos_y_rela=True, pos_z_rela=True, pos_z=True),
    Center(),
    GridSampling3D(grid_size=0.2, quantize_coords=True, mode=last),
    ShiftVoxels(apply_shift=True),
])
val_transform = Compose([
    XYZRelaFeature(axis=['x_rela', 'y_rela', 'z_rela']),
    XYZFeature(axis=['z']),
    AddFeatsByKeys(pos_x_rela=True, pos_y_rela=True, pos_z_rela=True, pos_z=True),
    Center(),
    GridSampling3D(grid_size=0.2, quantize_coords=True, mode=last),
])
inference_transform = Compose([
    XYZRelaFeature(axis=['x_rela', 'y_rela', 'z_rela']),
    XYZFeature(axis=['z']),
    AddFeatsByKeys(pos_x_rela=True, pos_y_rela=True, pos_z_rela=True, pos_z=True),
    Center(),
    GridSampling3D(grid_size=0.2, quantize_coords=True, mode=last),
])
pre_collate_transform = Compose([
    SaveOriginalPosId,
    GridSampling3D(grid_size=0.2, quantize_coords=False, mode=last),
])
Size of train_dataset = 3000
Size of test_dataset = 319
Size of val_dataset = 129
Batch size = 4
[2026-07-21 12:16:35,556][torch_points3d.datasets.base_dataset][INFO] - Available stage selection datasets:  ['test', 'val'] 
[2026-07-21 12:16:35,558][torch_points3d.datasets.base_dataset][INFO] - The models will be selected using the metrics on following dataset:  val 
[2026-07-21 12:16:41,105][torch_points3d.trainer][INFO] - EPOCH 1 / 10
100%|█| 750/750 [17:16<00:00,  1.38s/it, data_loading=1.359, iteration=0.503, train_F1=0.0  , train_Iacc=0.0  , train_acc=75.12, train_cov=0.0  , train_ins_dist_loss=0.835, train_ins_loss=1.082, train_ins_reg_los
[2026-07-21 12:34:00,780][torch_points3d.trainer][INFO] - Learning rate = 0.000989
100%|█| 33/33 [00:25<00:00,  1.31it/s, val_F1=0.0  , val_Iacc=0.0  , val_acc=85.38, val_cov=0.0  , val_ins_dist_loss=0.199, val_ins_loss=0.409, val_ins_reg_loss=0.004, val_ins_var_loss=0.206, val_loss=0.993, val_
[2026-07-21 12:34:29,627][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 12:34:29,628][torch_points3d.metrics.base_tracker][INFO] -     val_loss = 0.992956280708313
[2026-07-21 12:34:29,629][torch_points3d.metrics.base_tracker][INFO] -     val_offset_norm_loss = 2.6269493103027344
[2026-07-21 12:34:29,630][torch_points3d.metrics.base_tracker][INFO] -     val_offset_dir_loss = -0.7504850029945374
[2026-07-21 12:34:29,631][torch_points3d.metrics.base_tracker][INFO] -     val_ins_loss = 0.40961167216300964
[2026-07-21 12:34:29,632][torch_points3d.metrics.base_tracker][INFO] -     val_ins_var_loss = 0.2061537504196167
[2026-07-21 12:34:29,633][torch_points3d.metrics.base_tracker][INFO] -     val_ins_dist_loss = 0.19924280047416687
[2026-07-21 12:34:29,634][torch_points3d.metrics.base_tracker][INFO] -     val_ins_reg_loss = 0.0042151035740971565
[2026-07-21 12:34:29,635][torch_points3d.metrics.base_tracker][INFO] -     val_semantic_loss = 0.3956981897354126
[2026-07-21 12:34:29,637][torch_points3d.metrics.base_tracker][INFO] -     val_acc = 85.38136751949772
[2026-07-21 12:34:29,638][torch_points3d.metrics.base_tracker][INFO] -     val_macc = 53.077682200236694
[2026-07-21 12:34:29,639][torch_points3d.metrics.base_tracker][INFO] -     val_miou = 44.996615468482084
[2026-07-21 12:34:29,641][torch_points3d.metrics.base_tracker][INFO] -     val_iou_per_class = {0: '61.73', 1: '24.77', 2: '46.44', 3: '86.42', 4: '5.62'}
[2026-07-21 12:34:29,641][torch_points3d.metrics.base_tracker][INFO] -     val_pos = 0.0
[2026-07-21 12:34:29,642][torch_points3d.metrics.base_tracker][INFO] -     val_neg = 0.0
[2026-07-21 12:34:29,644][torch_points3d.metrics.base_tracker][INFO] -     val_Iacc = 0.0
[2026-07-21 12:34:29,645][torch_points3d.metrics.base_tracker][INFO] -     val_cov = 0.0
[2026-07-21 12:34:29,646][torch_points3d.metrics.base_tracker][INFO] -     val_wcov = 0.0
[2026-07-21 12:34:29,647][torch_points3d.metrics.base_tracker][INFO] -     val_mIPre = 0.0
[2026-07-21 12:34:29,648][torch_points3d.metrics.base_tracker][INFO] -     val_mIRec = 0.0
[2026-07-21 12:34:29,649][torch_points3d.metrics.base_tracker][INFO] -     val_F1 = 0.0
[2026-07-21 12:34:29,650][torch_points3d.metrics.base_tracker][INFO] - ==================================================
100%|█| 80/80 [01:04<00:00,  1.23it/s, test_F1=0.0  , test_Iacc=0.0  , test_acc=85.20, test_cov=0.0  , test_ins_dist_loss=0.199, test_ins_loss=0.409, test_ins_reg_loss=0.004, test_ins_var_loss=0.206, test_loss=0.
[2026-07-21 12:35:38,450][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 12:35:38,451][torch_points3d.metrics.base_tracker][INFO] -     test_loss = 0.992956280708313
[2026-07-21 12:35:38,452][torch_points3d.metrics.base_tracker][INFO] -     test_offset_norm_loss = 2.6269493103027344
[2026-07-21 12:35:38,454][torch_points3d.metrics.base_tracker][INFO] -     test_offset_dir_loss = -0.7504850029945374
[2026-07-21 12:35:38,455][torch_points3d.metrics.base_tracker][INFO] -     test_ins_loss = 0.40961167216300964
[2026-07-21 12:35:38,456][torch_points3d.metrics.base_tracker][INFO] -     test_ins_var_loss = 0.2061537504196167
[2026-07-21 12:35:38,457][torch_points3d.metrics.base_tracker][INFO] -     test_ins_dist_loss = 0.19924280047416687
[2026-07-21 12:35:38,458][torch_points3d.metrics.base_tracker][INFO] -     test_ins_reg_loss = 0.0042151035740971565
[2026-07-21 12:35:38,459][torch_points3d.metrics.base_tracker][INFO] -     test_semantic_loss = 0.3956981897354126
[2026-07-21 12:35:38,460][torch_points3d.metrics.base_tracker][INFO] -     test_acc = 85.20894745968441
[2026-07-21 12:35:38,461][torch_points3d.metrics.base_tracker][INFO] -     test_macc = 53.98120867524653
[2026-07-21 12:35:38,462][torch_points3d.metrics.base_tracker][INFO] -     test_miou = 44.92657844951104
[2026-07-21 12:35:38,464][torch_points3d.metrics.base_tracker][INFO] -     test_iou_per_class = {0: '60.38', 1: '31.15', 2: '43.29', 3: '86.61', 4: '3.21'}
[2026-07-21 12:35:38,465][torch_points3d.metrics.base_tracker][INFO] -     test_pos = 0.0
[2026-07-21 12:35:38,466][torch_points3d.metrics.base_tracker][INFO] -     test_neg = 0.0
[2026-07-21 12:35:38,466][torch_points3d.metrics.base_tracker][INFO] -     test_Iacc = 0.0
[2026-07-21 12:35:38,467][torch_points3d.metrics.base_tracker][INFO] -     test_cov = 0.0
[2026-07-21 12:35:38,468][torch_points3d.metrics.base_tracker][INFO] -     test_wcov = 0.0
[2026-07-21 12:35:38,469][torch_points3d.metrics.base_tracker][INFO] -     test_mIPre = 0.0
[2026-07-21 12:35:38,470][torch_points3d.metrics.base_tracker][INFO] -     test_mIRec = 0.0
[2026-07-21 12:35:38,471][torch_points3d.metrics.base_tracker][INFO] -     test_F1 = 0.0
[2026-07-21 12:35:38,472][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 12:35:38,473][torch_points3d.trainer][INFO] - EPOCH 2 / 10
100%|█| 750/750 [1:35:49<00:00,  7.67s/it, data_loading=0.472, iteration=0.331, train_F1=0.0  , train_Iacc=0.0  , train_acc=87.30, train_cov=0.0  , train_ins_dist_loss=0.181, train_ins_loss=0.337, train_ins_reg_l
[2026-07-21 14:11:31,857][torch_points3d.trainer][INFO] - Learning rate = 0.000977
100%|█| 33/33 [00:14<00:00,  2.32it/s, val_F1=0.0  , val_Iacc=0.0  , val_acc=86.84, val_cov=0.0  , val_ins_dist_loss=0.240, val_ins_loss=0.388, val_ins_reg_loss=0.004, val_ins_var_loss=0.143, val_loss=0.864, val_
[2026-07-21 14:11:46,260][torch_points3d.utils.colors][INFO] - loss: 0.992956280708313 -> 0.864528238773346, offset_norm_loss: 2.6269493103027344 -> 2.2270076274871826, offset_dir_loss: -0.7504850029945374 -> -0.7831555008888245, ins_loss: 0.40961167216300964 -> 0.3885045051574707, ins_var_loss: 0.2061537504196167 -> 0.1438693404197693, ins_reg_loss: 0.0042151035740971565 -> 0.0041654035449028015, semantic_loss: 0.3956981897354126 -> 0.33163854479789734, acc: 85.38136751949772 -> 86.84691502151387, macc: 53.077682200236694 -> 63.72215231738435, miou: 44.996615468482084 -> 53.21738586982049, pos: 0.0 -> 0.0, neg: 0.0 -> 0.0, Iacc: 0.0 -> 0.0, cov: 0.0 -> 0.0, wcov: 0.0 -> 0.0, mIPre: 0.0 -> 0.0, mIRec: 0.0 -> 0.0, F1: 0.0 -> 0.0
[2026-07-21 14:11:50,704][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 14:11:50,705][torch_points3d.metrics.base_tracker][INFO] -     val_loss = 0.864528238773346
[2026-07-21 14:11:50,706][torch_points3d.metrics.base_tracker][INFO] -     val_offset_norm_loss = 2.2270076274871826
[2026-07-21 14:11:50,707][torch_points3d.metrics.base_tracker][INFO] -     val_offset_dir_loss = -0.7831555008888245
[2026-07-21 14:11:50,708][torch_points3d.metrics.base_tracker][INFO] -     val_ins_loss = 0.3885045051574707
[2026-07-21 14:11:50,709][torch_points3d.metrics.base_tracker][INFO] -     val_ins_var_loss = 0.1438693404197693
[2026-07-21 14:11:50,711][torch_points3d.metrics.base_tracker][INFO] -     val_ins_dist_loss = 0.24046975374221802
[2026-07-21 14:11:50,712][torch_points3d.metrics.base_tracker][INFO] -     val_ins_reg_loss = 0.0041654035449028015
[2026-07-21 14:11:50,713][torch_points3d.metrics.base_tracker][INFO] -     val_semantic_loss = 0.33163854479789734
[2026-07-21 14:11:50,715][torch_points3d.metrics.base_tracker][INFO] -     val_acc = 86.84691502151387
[2026-07-21 14:11:50,716][torch_points3d.metrics.base_tracker][INFO] -     val_macc = 63.72215231738435
[2026-07-21 14:11:50,718][torch_points3d.metrics.base_tracker][INFO] -     val_miou = 53.21738586982049
[2026-07-21 14:11:50,720][torch_points3d.metrics.base_tracker][INFO] -     val_iou_per_class = {0: '66.17', 1: '24.88', 2: '51.75', 3: '87.70', 4: '35.59'}
[2026-07-21 14:11:50,722][torch_points3d.metrics.base_tracker][INFO] -     val_pos = 0.0
[2026-07-21 14:11:50,723][torch_points3d.metrics.base_tracker][INFO] -     val_neg = 0.0
[2026-07-21 14:11:50,725][torch_points3d.metrics.base_tracker][INFO] -     val_Iacc = 0.0
[2026-07-21 14:11:50,726][torch_points3d.metrics.base_tracker][INFO] -     val_cov = 0.0
[2026-07-21 14:11:50,728][torch_points3d.metrics.base_tracker][INFO] -     val_wcov = 0.0
[2026-07-21 14:11:50,730][torch_points3d.metrics.base_tracker][INFO] -     val_mIPre = 0.0
[2026-07-21 14:11:50,732][torch_points3d.metrics.base_tracker][INFO] -     val_mIRec = 0.0
[2026-07-21 14:11:50,733][torch_points3d.metrics.base_tracker][INFO] -     val_F1 = 0.0
[2026-07-21 14:11:50,735][torch_points3d.metrics.base_tracker][INFO] - ==================================================
100%|█| 80/80 [00:40<00:00,  1.98it/s, test_F1=0.0  , test_Iacc=0.0  , test_acc=86.00, test_cov=0.0  , test_ins_dist_loss=0.240, test_ins_loss=0.388, test_ins_reg_loss=0.004, test_ins_var_loss=0.143, test_loss=0.
[2026-07-21 14:12:35,894][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 14:12:35,895][torch_points3d.metrics.base_tracker][INFO] -     test_loss = 0.864528238773346
[2026-07-21 14:12:35,896][torch_points3d.metrics.base_tracker][INFO] -     test_offset_norm_loss = 2.2270076274871826
[2026-07-21 14:12:35,898][torch_points3d.metrics.base_tracker][INFO] -     test_offset_dir_loss = -0.7831555008888245
[2026-07-21 14:12:35,899][torch_points3d.metrics.base_tracker][INFO] -     test_ins_loss = 0.3885045051574707
[2026-07-21 14:12:35,900][torch_points3d.metrics.base_tracker][INFO] -     test_ins_var_loss = 0.1438693404197693
[2026-07-21 14:12:35,900][torch_points3d.metrics.base_tracker][INFO] -     test_ins_dist_loss = 0.24046975374221802
[2026-07-21 14:12:35,902][torch_points3d.metrics.base_tracker][INFO] -     test_ins_reg_loss = 0.0041654035449028015
[2026-07-21 14:12:35,903][torch_points3d.metrics.base_tracker][INFO] -     test_semantic_loss = 0.33163854479789734
[2026-07-21 14:12:35,904][torch_points3d.metrics.base_tracker][INFO] -     test_acc = 86.0049013640189
[2026-07-21 14:12:35,905][torch_points3d.metrics.base_tracker][INFO] -     test_macc = 61.931322385677575
[2026-07-21 14:12:35,906][torch_points3d.metrics.base_tracker][INFO] -     test_miou = 51.623688781009456
[2026-07-21 14:12:35,907][torch_points3d.metrics.base_tracker][INFO] -     test_iou_per_class = {0: '61.46', 1: '33.91', 2: '48.76', 3: '87.08', 4: '26.90'}
[2026-07-21 14:12:35,908][torch_points3d.metrics.base_tracker][INFO] -     test_pos = 0.0
[2026-07-21 14:12:35,909][torch_points3d.metrics.base_tracker][INFO] -     test_neg = 0.0
[2026-07-21 14:12:35,910][torch_points3d.metrics.base_tracker][INFO] -     test_Iacc = 0.0
[2026-07-21 14:12:35,911][torch_points3d.metrics.base_tracker][INFO] -     test_cov = 0.0
[2026-07-21 14:12:35,912][torch_points3d.metrics.base_tracker][INFO] -     test_wcov = 0.0
[2026-07-21 14:12:35,913][torch_points3d.metrics.base_tracker][INFO] -     test_mIPre = 0.0
[2026-07-21 14:12:35,914][torch_points3d.metrics.base_tracker][INFO] -     test_mIRec = 0.0
[2026-07-21 14:12:35,915][torch_points3d.metrics.base_tracker][INFO] -     test_F1 = 0.0
[2026-07-21 14:12:35,915][torch_points3d.metrics.base_tracker][INFO] - ==================================================
[2026-07-21 14:12:35,917][torch_points3d.trainer][INFO] - EPOCH 3 / 10
 60%|▌| 447/750 [10:21<06:22,  1.26s/it, data_loading=0.531, iteration=0.429, train_F1=0.0  , train_Iacc=0.0  , train_acc=86.51, train_cov=0.0  , train_ins_dist_loss=0.162, train_ins_loss=0.316, train_ins_reg_los
```