
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
    num_inputs: usize,
    num_outputs: usize,

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

    pub fn eval_layer(&self, inputs: &Vec<T>) -> Vec<T> {
        if self.num_inputs != inputs.len() {
            panic!("incorrect number of inputs to layer");
        }

        let propagated = self.weights.multiply_vec(inputs);
        return propagated.into_iter().map(sigmod).collect();
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
    activations: Vec<Vec<T>>,
    sigmas: Vec<Vec<T>>
}

impl<T: Float> Network<T> {
    pub fn new() -> Network<T> {
        panic!("Implement me");
    }

    pub fn set_input_activations(input: &Vec<T>) {
        panic!("Implement me");
    }

    pub fn propagate() {
        panic!("Implement me");
    }

    pub fn back_propagate() {
        panic!("Implement me");
    }
}

