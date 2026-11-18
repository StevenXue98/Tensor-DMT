from cookie_problem import CookieExperimentConfig, run_cookie_problem


def main() -> None:
    config = CookieExperimentConfig(
        sizes=(8,),
        repeats=1,
        tol=1e-5,
        rounding_tol=1e-7,
        max_iter=20,
        oversampling=5,
        seed=12,
    )

    records = run_cookie_problem(config)

    for record in records:
        print(
            f"n={record.n}, "
            f"method={record.method}, "
            f"time={record.elapsed:.3f}s, "
            f"residual={record.residual:.3e}, "
            f"final rank={record.ranks[-1]}"
        )


if __name__ == "__main__":
    main()