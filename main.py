import argparse

from training import text_pretrain, train_hdc_mlp
from inference import lhdm_infer

def main():
    parser = argparse.ArgumentParser(
        description="LHDM: text pretraining, visual training, and inference."
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["text_pretrain", "train_visual", "infer"],
        help="Which stage to run.",
    )

    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="data/clinical_prompts_10k.csv",
        help="CSV file with clinical prompts for text pretraining.",
    )
    parser.add_argument(
        "--zhang_root",
        type=str,
        default="data/Zhang",
        help="Root folder for Zhang OCT dataset.",
    )
    parser.add_argument(
        "--duke_root",
        type=str,
        default="data/Duke",
        help="Root folder for Duke OCT dataset.",
    )
    parser.add_argument(
        "--rabbani_root",
        type=str,
        default="data/Rabbani",
        help="Root folder for Rabbani OCT dataset.",
    )
    parser.add_argument(
        "--biomisa_root",
        type=str,
        default="data/BIOMISA",
        help="Root folder for BIOMISA OCT dataset.",
    )

    args = parser.parse_args()

    if args.stage == "text_pretrain":
        # Run text pretraining, passing prompts_csv override
        text_pretrain.main(csv_path=args.prompts_csv)

    elif args.stage == "train_visual":
        # Train HDC+MLP on Zhang dataset (root override)
        train_hdc_mlp.main(zhang_root=args.zhang_root)

    elif args.stage == "infer":
        # Run LHDM inference on all datasets whose roots exist
        lhdm_infer.main(
            zhang_root=args.zhang_root,
            duke_root=args.duke_root,
            rabbani_root=args.rabbani_root,
            biomisa_root=args.biomisa_root,
        )


if __name__ == "__main__":
    main()
