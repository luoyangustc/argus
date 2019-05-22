# Shadow

## Required
1. Install Google's Protobuf

    For Ubuntu user: `apt install libprotobuf-dev protobuf-compiler`.

    For Mac user: `brew install protobuf`.

## Installation

1. Run `./scripts/build_shell.sh` to build the whole project.

## Demo

1. Run `./data/get_data.sh` to get static image samples.

2. Run `./models/get_models.sh` to get some test models.

3. To run the demo, run

    ```
    ./build/shadow/test_inference
    ```

