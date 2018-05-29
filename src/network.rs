use linear::Matrix;
use std::ops::AddAssign;
use std::fmt::Display;
use real::sigmoid_prime;
use num_traits::{Float, NumCast};
use rand::{Rng, Rand};
use linear::print_vector;

pub fn sigmoid<T: Float + NumCast>(x: T) -> T {
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

    pub fn eval_layer(&self, activations: &Vec<T>, z: &mut Vec<T>, activations_next: &mut Vec<T>) {
        println!("eval_layer activations: {}, z: {}, activations_next: {}",
                 activations.len(), z.len(), activations_next.len());
        if self.num_inputs != activations.len() {
            panic!("activations was wrong length. activations.len() = {}, num_inputs = {}",
                   activations.len(), self.num_inputs);
        }
        if self.num_outputs != z.len() {
            panic!("z was wrong length. z.len() = {}, self.num_outputs = {}", z.len(), self.num_outputs);
        }
        if self.num_outputs != activations_next.len() {
            panic!("incorrect number of outputs to layer");
        }

        self.weights.multiply_vec(activations, z);
        for i in 0..z.len() {
            z[i] = z[i] + self.biases[i];
            print!("z="); print_vector(z); println!();
            activations_next[i] = sigmoid(z[i]);
            print!("a="); print_vector(activations_next); println!();
        }
    }

    pub fn find_output_sigma(&self,
                             activations: &Vec<T>,
                             z: &Vec<T>,
                             expected: &Vec<T>,
                             activation_errors: &mut Vec<T>) {
        //Compute the output layer error
        for k in 0..self.num_outputs {
            let d_cost_by_d_activation_k: T = T::from(2.0).unwrap()
                * (activations[k] - expected[k]);
            let exp_z = Float::exp(-z[k]);
            let one_minus_exp_z = T::one() - exp_z;
            let d_activation_by_d_z = -exp_z / one_minus_exp_z / one_minus_exp_z;

            println!("{} {}", d_cost_by_d_activation_k, d_activation_by_d_z);
            activation_errors[k] = d_cost_by_d_activation_k * d_activation_by_d_z;
        }
    }

    pub fn back_propagate_layer(&self,
                                z: &Vec<T>,
                                activation_errors: &Vec<T>,
                                prev_activation_errors: &mut Vec<T>) {
        //j iterates the input layer, k iterates the output layer
        println!("Back propagate_layer. z={}, activation_errors={}, prev_activation_errors={}",
                 z.len(), activation_errors.len(), prev_activation_errors.len());

        for j in 0..self.num_inputs {
            let mut total = T::zero();
            for k in 0..self.num_outputs {
                //TODO: We are calculating sigmoid_prime too many times here
                let sigmoid_prime_of_z_l_j = sigmoid_prime(z[k]);
                total += self.weights.get(k, j) * activation_errors[k];
            }
            prev_activation_errors[j] = total;
        }
    }
}


pub struct Network<T> {
    pub num_inputs: usize,
    pub num_outputs: usize,

    pub layers: Vec<NetworkLayer<T>>,
    pub z: Vec<Vec<T>>,
    pub activations: Vec<Vec<T>>,
    pub activation_errors: Vec<Vec<T>>,
}

impl<T: Float + Display + AddAssign + Rand> Network<T> {
    pub fn new(layers: Vec<NetworkLayer<T>>) -> Network<T> {
        let layer_count = layers.len();
        let num_inputs = layers[0].num_inputs;
        let num_outputs = layers[layer_count - 1].num_outputs;

        let mut activations = Vec::<Vec<T>>::with_capacity(layers.len() + 1);
        let mut activation_errors = Vec::<Vec<T>>::with_capacity(layers.len() + 1);
        let mut z = Vec::<Vec<T>>::with_capacity(layers.len());

        activations.push(vec![T::zero(); layers[0].num_inputs]);
        activation_errors.push(vec![T::zero(); layers[0].num_inputs]);

        for layer in layers.iter() {
            activations.push(vec![T::zero(); layer.num_outputs]);
            activation_errors.push(vec![T::zero(); layer.num_outputs]);
            z.push(vec![T::zero(); layer.num_outputs]);
        }

        return Network {
            num_inputs,
            num_outputs,

            layers,
            z,
            activations,
            activation_errors,
        };
    }

    pub fn describe(&self) {
        println!("Network with the following layers");
        for i in 0..self.layers.len() {
            println!("{} {} -> {}", i, self.layers[i].num_inputs, self.layers[i].num_outputs);
        }
    }

    pub fn set_input_activations(&mut self, input: &Vec<T>) {
        if input.len() != self.num_inputs {
            panic!("wrong length of input activations");
        }
        println!("self.activations[0].len() = {}, input.len() = {}", self.activations[0].len(), input.len());
        self.activations[0].copy_from_slice(input.as_slice());
    }

    pub fn propagate(&mut self) {
        for i in 0..self.layers.len() {
            println!("propagate {}", i);
            let (front, back) = self.activations.split_at_mut(i + 1);
            let this_activations = &front[i];
            let mut next_activations = &mut back[0];
            self.layers[i].eval_layer(&this_activations,
                                      &mut self.z[i],
                                      &mut next_activations);
        }
    }

    pub fn back_propagate(&mut self, expected: &Vec<T>) {
        let last_layer = self.layers.len() - 1;

        self.layers[last_layer].find_output_sigma(
            &self.activations[last_layer],
            &self.z[last_layer],
            expected,
            &mut self.activation_errors[last_layer],
        );

        self.layers[self.layers.len() - 1].find_output_sigma(
            &self.activations[last_layer],
            &self.z[last_layer],
            expected,
            &mut self.activation_errors[last_layer],
        );

        for i in (0..self.layers.len()).rev() {
            println!("back_propagate, i={}", i);
            let (front, back) = self.activation_errors.split_at_mut(i + 1);
            let mut prev_activation_errors = &mut front[i];
            let this_activation_errors = &back[0];

            self.layers[i].back_propagate_layer(
                &self.z[i],
                this_activation_errors,
                prev_activation_errors,
            );
        }
    }

    pub fn find_d_cost_by_final_activations(&mut self, expected: &Vec<T>) {
        if expected.len() != self.num_outputs {
            panic!("expected.len() != num_outputs ");
        }
        for i in 0..self.num_outputs {
            self.activation_errors[self.layers.len()][i] =
                T::from(2).unwrap() * (expected[i] - self.activations[self.layers.len()][i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use network::sigmoid;

    #[test]
    fn test_sigmoid() {
        assert_approx_eq!(sigmoid(0.0f32), 0.5f32);
    }
}
