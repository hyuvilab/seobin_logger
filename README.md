# seobin_logger


An integrated logger for machine learning..


## Installation

```console
foo@bar:~/seobin_logger $ pip install .
```

## Usage


```python
import seobin_logger


train_logger = seobin_logger.MainLogger(['state1', 'state2'])
train_logger <= seobin_logger.TQDMLogger()
train_logger.start()

for i in range(10):
    train_logger.step({
        'state1': 1.,
        'state2': 2.,
    })
```

* Simple example shown in [simple_example.py](examples/simple_example.py)
