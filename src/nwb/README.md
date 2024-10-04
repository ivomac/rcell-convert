# NWB

#### A python module to validate, create, and load NWB files, and parse stimuli.

## How to use

The module is meant to be imported and used in other scripts.

After installation of the package, import the module as:

```python
import nwb
```

The main entry points are the load and save functions, and the cellDB and stimulus classes:

```python
id = "qpc_000000_20"
path = cellDB.get_path(id)

cell_dict = {
    "acquisition": { ... },
    "stimulus": { ... },
    ...
}

nwb.save(path, cell_dict, validate=True)
```

```python
id = "qpc_000000_20"
rcell = cellDB.load(id)


for rep in rcell.protocol("Activation"):
    rep.plot()
```

See the shared notebooks for more examples.

## Environment variables

Set the following environment variables in a .env file to use the module:

* STIMULUS_PATH:                   path to the folder containing files related to the stimuli.
* {IGOR,QPC,SYNCROPATCH}_NWB_PATH: paths where the NWB files are saved.

