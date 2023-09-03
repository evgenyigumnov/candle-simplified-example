# candle-simplified-example
A simplified example in Rust of training a neural network and then using it based on the [Candle Framework](https://github.com/huggingface/candle) by [Hugging Face](https://huggingface.co/).
## How its works

This program implements a neural network to predict the winner of the second round of elections based on the results of the first round.

Basic moments:

1. A multilayer perceptron with two hidden layers is used. The first hidden layer has 4 neurons, the second has 2 neurons.
2. The input is a vector of 2 numbers - the percentage of votes for the first and second candidates in the first stage.
3. The output is the number 0 or 1, where 1 means that the first candidate will win in the second stage, 0 means that he will lose.
4. For training, samples with real data on the results of the first and second stages of different elections are used.
5. The model is trained by backpropagation using gradient descent and the cross-entropy loss function.
6. Model parameters (weights of neurons) are initialized randomly, then optimized during training.
7. After training, the model is tested on a deferred sample to evaluate the accuracy.
8. If the accuracy on the test set is below 100%, the model is considered underfit and the learning process is repeated. 

Thus, this neural network learns to find hidden relationships between the results of the first and second rounds of voting in order to make predictions for new data.

## What does the code look like

```rust
const VOTE_DIM: usize = 2;
const RESULTS: usize = 1;
const EPOCHS: usize = 20;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 2;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_votes: Tensor,
    pub train_results: Tensor,
    pub test_votes: Tensor,
    pub test_results: Tensor,
}

struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}


pub fn main() -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let train_votes_vec: Vec<u32> = vec![
        15, 10,
        10, 15,
        5, 12,
        30, 20,
        16, 12,
        13, 25,
        6, 14,
        31, 21,
    ];
    let train_votes_tensor = Tensor::from_vec(train_votes_vec.clone(), (train_votes_vec.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let train_results_vec: Vec<u32> = vec![
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
    ];
    let train_results_tensor = Tensor::from_vec(train_results_vec, train_votes_vec.len() / VOTE_DIM, &dev)?;

    let test_votes_vec: Vec<u32> = vec![
        13, 9,
        8, 14,
        3, 10,
    ];
    let test_votes_tensor = Tensor::from_vec(test_votes_vec.clone(), (test_votes_vec.len() / VOTE_DIM, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let test_results_vec: Vec<u32> = vec![
        1,
        0,
        0,
    ];
    let test_results_tensor = Tensor::from_vec(test_results_vec.clone(), test_results_vec.len(), &dev)?;

    let m = Dataset {
        train_votes: train_votes_tensor,
        train_results: train_results_tensor,
        test_votes: test_votes_tensor,
        test_results: test_results_tensor,
    };

    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            },
            Err(e) => {
                println!("Error: {:?}", e);
                continue;
            }
        }

    }

    let real_world_votes: Vec<u32> = vec![
        13, 22,
    ];

    let tensor_test_votes = Tensor::from_vec(real_world_votes.clone(), (1, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let final_result = trained_model.forward(&tensor_test_votes)?;

    let result = final_result
        .argmax(D::Minus1)?
        .to_dtype(DType::F32)?
        .get(0).map(|x| x.to_scalar::<f32>())??;
    println!("real_life_votes: {:?}", real_world_votes);
    println!("neural_network_prediction_result: {:?}", result);

    Ok(())
}
```

## How to run

cargo run

## Example output

```

Trying to train neural network.
Epoch:   1 Train loss:  4.42555 Test accuracy:  0.00%
Epoch:   2 Train loss:  0.84677 Test accuracy: 33.33%
Epoch:   3 Train loss:  2.54335 Test accuracy: 33.33%
Epoch:   4 Train loss:  0.37806 Test accuracy: 33.33%
Epoch:   5 Train loss:  0.36647 Test accuracy: 100.00%
real_life_votes: [13, 22]
neural_network_prediction_result: 0.0

```