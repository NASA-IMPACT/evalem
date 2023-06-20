# evalem
Machine Learning pipeline evaluation framework


# Contributing

We use [Hatch](https://hatch.pypa.io/latest/install/) for environment management and packaging `evalem`.

To start the development env, you'll first need to install hatch. We recommend installing with [pipx](https://github.com/pypa/pipx) so it doesn't interfere with other python envs.

```
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

This globally installs hatch, restart the terminal for changes to take effect. Then,

`pipx install hatch` 

With hatch installed, you can start a shell environment with test dependencies, creating the env if it doesn't already exist.

`hatch -e test shell`

See the pyproject.toml for the test dependencies. The test environment inherits dependencies from the default environment.

If you need to remove dependencies, you'll need to delete the environments and recreate them. Use 

`hatch env prune` to remove all environments and `hatch -e test shell' to recreate the default and test environments.



