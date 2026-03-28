from src.inference.model_loader import load_model
from src.utils.config_loader import load_config
from src.inference.generate import generate_text


def main():
    config = load_config("configs.py/base.yaml")

    model, tokenizer = load_model(
        config["model"]["name"],
        config["model"]["device"]

    )


    prompt = "Explain what is Lora in simple terms"

    output = generate_text(model, tokenizer, prompt, config)

    print("\n===OUTPUT===\n")
    print(output)


if __name__ =="__main__":
    main()