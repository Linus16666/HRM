import subprocess
import sys


def main():
    args = []
    it = iter(sys.argv[1:])
    for arg in it:
        if arg.startswith("--"):
            arg = arg[2:]
            if "=" not in arg:
                try:
                    value = next(it)
                except StopIteration:
                    continue
                args.append(f"{arg}={value}")
            else:
                args.append(arg)
        else:
            args.append(arg)
    subprocess.run([sys.executable, "pretrain.py", *args], check=True)


if __name__ == "__main__":
    main()
