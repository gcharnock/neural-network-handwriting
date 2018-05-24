
use linear::Matrix;
use std::ops::AddAssign;
use std::fmt::Display;
use real::sigmoid_prime;
use num_traits::{Float, NumCast};
use rand::{Rng, Rand};

pub fn sigmod<T: Float + NumCast>(x: T) -> T {
    let one: T = NumCast::from(1).unwrap();
    one / (one + Float::exp(-x))
}

pub struct NetworkLayer<T> {
    pub num_inputs: usize,
    pub num_outputs: usize,

    weights: Matrix<T>,
    biases: Vec<T>,
}



impl<T: Clone + AddAssign + Display + Rand + Float + NumCast> NetworkLayer<T>
    where {
    pub fn new<R: Rng>(num_inputs: usize, num_outputs: usize, gen: &mut R) -> NetworkLayer<T> {
        let mut gen_closure = || {
            let r: T = Rand::rand(gen);
            r - NumCast::from(0.5).unwrap()
        };
        return NetworkLayer {
            num_inputs,
            num_outputs,

            weights: Matrix::new(num_outputs, num_inputs, &mut gen_closure),
            biases: vec![T::zero(); num_inputs],
        };
    }

    pub fn eval_layer(&self, inputs: &Vec<T>, outputs: &mut Vec<T>) {
        if self.num_inputs != inputs.len() {
            panic!("incorrect number of inputs to layer");
        }
        if self.num_outputs != outputs.len() {
            panic!("incorrect number of inputs to layer");
        }

        let propagated = vec![T::zero(), self.num_outputs];
        self.weights.multiply_vec(inputs, &mut propagated);

        return propagated.into_iter().map(|x|outputs(sigmod(x)));
    }

    pub fn find_output_sigma(&self, output: &Vec<T>, expected: &Vec<T>) -> Vec<T> {
        //Compute the output layer error
        let mut sigma_last = Vec::with_capacity(self.num_outputs);
        for k in 0..self.num_outputs {
            let exp_x = Float::exp(-output[k]);
            let one_minus_exp_x = T::one() - exp_x;
            let sigma_last_k = -exp_x / one_minus_exp_x / one_minus_exp_x;
            sigma_last.push(sigma_last_k);
        }
        return sigma_last;
    }

    pub fn back_propagate_layer(&self, input: &Vec<T>, sigma_l_plus_1: &Vec<T>, expected: &Vec<T>) -> Vec<T> {
        //j iterates the input layer, k iterates the output layer

        let mut sigma_l = Vec::<T>::with_capacity(self.num_inputs);
        for j in 0..self.num_inputs {
            let mut total = T::zero();
            for k in 0..self.num_outputs {
                total += self.weights.get(k, j) * sigma_l_plus_1[k];
            }
            let sigmoid_prime_of_z_l_j = sigmoid_prime(input[j]);
            sigma_l.push(total);
        }
        return sigma_l;
    }
}


pub struct Network<T> {
    num_inputs: usize,
    num_outputs: usize,

    layers: Vec<NetworkLayer<T>>,
    z: Vec<Vec<T>>,
    activations: Vec<Vec<T>>,
    activation_errors: Vec<Vec<T>>
}

impl<T: Float> Network<T> {
    pub fn new(layers: Vec<NetworkLayer<T>>) -> Network<T> {
        let layer_count = layers.len();
        let num_inputs = layers[0].num_inputs;
        let num_outputs = layers[layer_count - 1].num_outputs;

        let mut activations = Vec::with_capacity(layers.len() + 1);
        activations = layers.iter()
            .map(|layer|vec![0; layer.num_inputs])
            .collect();
        activations.push(vec![0, num_outputs]);

        let mut activation_errors = Vec::with_capacity(layers.len() + 1);
        activation_errors = layers.iter()
            .map(|layer|vec![0; layer.num_inputs])
            .collect();
        activation_errors.push(vec![0, num_outputs]);


        let mut z = Vec::with_capacity(layers.len() + 1);
        layers.iter()
            .map(|layer|vec![0; layer.num_inputs])
            .collect();
        z.push(vec![0, num_outputs]);

        z = layers.iter()
            .map(|layer|vec![0; layer.num_inputs])
            .collect();
        z.push(vec![0, num_outputs]);

        return Network {
            num_inputs,
            num_outputs,

            layers,
            z,
            activations,
            activation_errors
        }
    }

    pub fn set_input_activations(&mut self, input: &Vec<T>) {
        if input.len() != self.num_inputs {
           panic!("wrong length of input activations");
        }
        self.activations[0].copy_from_slice(input.as_slice());
    }

    pub fn propagate(&mut self) {
        for i in 0..self.layers.len() {
            let layer = self.layers[0];
            layer.eval_layer();
        }
    }

    pub fn back_propagate() {
        panic!("Implement me");
    }
}

