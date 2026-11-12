"""
A first, short training run on the pictures you have labelled so far.

It is a test that everything works, and it gives you a starting model that can
pre-place points on your next moments. The score will look better than it really
is: with only one moment, training and test pictures are near-copies.

Steps:
  1. drops pictures you left blank (a cheetah with no points would teach "no cheetah here")
  2. sets up training, starting from SuperAnimal-Quadruped (already knows four-legged animals)
  3. trains
  4. tests it and draws its guesses next to your labels

Usage (dlc environment):
    python scripts/training/train.py cheetah-dlc/config.yaml
    python scripts/training/train.py cheetah-dlc/config.yaml --epochs 100 --shuffle 1
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int, default=100, help="passes through all your pictures (default 100)")
    ap.add_argument("--shuffle", type=int, default=1, help="experiment number (default 1)")
    ap.add_argument("--batch-size", type=int, default=8, help="lower to 4 if you get 'CUDA out of memory'")
    ap.add_argument("--workers", type=int, default=4,
                    help="CPU helpers that load pictures in parallel while the GPU trains (default 4)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="test during training every N epochs (default 0: only once, at the end)")
    args = ap.parse_args()

    import deeplabcut
    from deeplabcut.modelzoo import build_weight_init

    config = str(Path(args.config).expanduser().absolute())

    print("\n1. Removing blank pictures from the label files")
    deeplabcut.dropunlabeledframes(config)

    print("\n2. Setting up training (downloads SuperAnimal weights the first time)")
    weight_init = build_weight_init(
        cfg=config, super_animal="superanimal_quadruped",
        model_name="hrnet_w32", detector_name=None, with_decoder=False,
    )
    deeplabcut.create_training_dataset(
        config, Shuffles=[args.shuffle], net_type="hrnet_w32", weight_init=weight_init,
    )

    print(f"\n3. Training for {args.epochs} epochs")
    updates = {
        # Load pictures in parallel: decoding big PNGs on one CPU thread was the bottleneck.
        "train_settings.dataloader_workers": args.workers,
        "train_settings.dataloader_pin_memory": True,
        # Testing full-size pictures mid-training can overflow the GPU's memory, and Windows
        # then quietly borrows normal RAM, which makes everything crawl. Test at the end instead.
        "runner.eval_interval": args.eval_every if args.eval_every > 0 else args.epochs + 1,
        # Slow the learning down near the end (at 80% and 95%) whatever the number of epochs;
        # this is the step that makes the model settle and become confident.
        "runner.scheduler.params.milestones": [int(args.epochs * 0.8), int(args.epochs * 0.95)],
    }
    deeplabcut.train_network(
        config, shuffle=args.shuffle, device="cuda:0", epochs=args.epochs,
        save_epochs=max(1, args.epochs // 4), batch_size=args.batch_size,
        pytorch_cfg_updates=updates,
    )

    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()  # hand the training memory back before testing

    print("\n4. Testing")
    deeplabcut.evaluate_network(config, shuffles=[args.shuffle], plotting=True, per_keypoint_evaluation=True)
    print("\nDone. Pictures with your labels (+) and the model's guesses (dots) are in "
          f"{Path(config).parent / 'evaluation-results-pytorch'}")


if __name__ == "__main__":
    main()