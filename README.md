# Cloning and updating #

```
git clone --recursive https://github.com/mattjj/kz-example.git
```

To pull updates from master, it may also be necessary to run a submodule update each time:

```
git pull origin master
git submodule update --init --recursive
```

# Running #

From a shell:

```
python example.py
```

or from within `ipython --pylab`:

```
run example.py
```

