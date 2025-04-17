"""Test the SimpleLinearRegression in an end to end test."""

import csv
import os
import json
from math import isclose
from typing import List, Tuple

import pytest

from rustlearn import NamedArray
from rustlearn.linear_model import LinearRegression, LinearRegressionReturn
from rustlearn.model_selection.simple import SimpleTrainTestSplit, TrainTestSplitReturn
from rustlearn.metrics.mean_absolute_error import mean_absolute_error
from rustlearn.metrics.r_2 import r_2


CURR_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURE_PATH = os.path.join(CURR_PATH, "fixtures")


@pytest.fixture
def input_data() -> Tuple[List[NamedArray], NamedArray]:
    """Load the input data from the static fixture."""
    data = {}
    with open(os.path.join(FIXTURE_PATH, "data.csv")) as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                for x in row:
                    data[x] = []
            else:
                keys = list(data.keys())
                for j, x in enumerate(row):
                    data[keys[j]].append(float(x))

    x_values = []
    for key, value in data.items():
        if key == "target":
            y_value = NamedArray(name="target", data=value)
        else:
            x_values.append(NamedArray(name=key, data=value))
    return x_values, y_value


def test_end_to_end(input_data: Tuple[List[NamedArray], NamedArray]) -> None:
    """Run the end to end test."""
    x = input_data[0]
    y = input_data[1]
    train_test_split = SimpleTrainTestSplit(x, y, 0.5)
    split_data = train_test_split.split()
    lin_reg = LinearRegression(x=split_data.x_train, y=split_data.y_train)
    ret = lin_reg.fit()
    y_pred = lin_reg.predict(new_x=split_data.x_test, return_object=ret)
    mae = mean_absolute_error(split_data.y_test, y_pred)
    r_squared = r_2(split_data.y_test, y_pred)

    assert isclose(mae, 43.18255622188293, rel_tol=0.001)
    assert isclose(r_squared, 0.5248892716360292, rel_tol=0.001)
