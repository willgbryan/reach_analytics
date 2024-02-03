import os
import argparse

from lats import run_lats


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--strategy", type=str,
                        help="Strategy: `simple`, `reflexion`")
    parser.add_argument("--language", type=str, help="Strategy: `py` or `rs`")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--expansion_factor", type=int,
                        help="The expansion factor for the reflexion UCS and A* strategy", default=3)
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    parser.add_argument("--instruction", type=str,
                        help="text string", default="")
    parser.add_argument("--n_samples", type=int,
                        help="The number of nodes added during expansion", default=3)
    parser.add_argument("--depth", type=int,
                        help="Tree depth", default=5)

    # TODO: implement this
    # parser.add_argument("--is_resume", action='store_true', help="To resume run")
    # parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    args = parser.parse_args()
    return args


def strategy_factory(strategy: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    return kwargs_wrapper_gen(run_lats, delete_keys=[])
    

def lats_main(args):

    # check if the strategy is valid
    run_strategy = strategy_factory(args.strategy)

    # start the run
    # evaluate with pass@k
    x = run_strategy(
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        verbose=args.verbose,
        instruction=args.instruction,
        n_samples=args.n_samples,
        depth=args.depth
    )

    return x



def main(args):
    lats_main(args)

if __name__ == "__main__":
    args = get_args()
    main(args)