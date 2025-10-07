# specbridge/cli.py
import sys
from specbridge.run import main as run_main

def main():
    # If the first token is our eval subcommand, route to it and strip the token.
    if len(sys.argv) > 1 and sys.argv[1] == "eval-retrieval":
        from specbridge.eval.retrieval import main as eval_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # drop the subcommand
        eval_main()
        return

    # Default: behave exactly like before (training / demo entry)
    run_main()

if __name__ == "__main__":
    main()
