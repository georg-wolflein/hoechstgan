# poetry run python train.py +experiment=hoechstgan_composite discriminator.type=joint generator.composites.0.train.schedule=sigmoid generator.composites.0.train.args.from_epoch=8 generator.composites.0.train.args.to_epoch=12 dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=hoechstgan_composite generator.composites.0.train.schedule=sigmoid generator.composites.0.train.args.from_epoch=8 generator.composites.0.train.args.to_epoch=12 dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=hoechstgan discriminator.type=joint dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=hoechstgan dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=hoechstgan_basic discriminator.type=joint dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=cy3 dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
poetry run python train.py +experiment=cy5 dataset.max_size=250000 norm=instance gpus=[2,3,4,5]
