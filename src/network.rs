use linear::Matrix;
use std::ops::AddAssign;
use std::fmt::Display;
use num_traits::{Float, NumCast};
use rand::{Rand, Rng};
use linear::print_vector;

pub fn sigmoid<T: Float>(x: T) -> T {
    let one: T = T::one();
    one / (one + Float::exp(-x))
}

pub fn sigmoid_vec<T: Float>(x: &Vec<T>, out: &mut Vec<T>) {
    for i in 0..x.len() {
        out[i] = sigmoid(x[i]);
    }
}

pub fn sigmoid_deriv<T: Float>(x: T) -> T {
    let exp_x = Float::exp(-x);
    let one_plus_exp_x = T::one() + exp_x;
    exp_x / one_plus_exp_x / one_plus_exp_x
}

fn sum_of_squared_error<T: Float>(label: u8, output: &Vec<T>) -> T {
    let label = label as usize;
    let mut total = T::zero();
    for i in 0..output.len() {
        if i == label {
            total = total + (output[i] - T::one()) * (output[i] - T::one());
        } else {
            total = total + output[i] * output[i];
        }
    }
    return total;
}

pub struct NetworkLayer<T> {
    pub num_inputs: usize,
    pub num_outputs: usize,

    weights: Matrix<T>,
    biases: Vec<T>,
}


impl<T: Clone + AddAssign + Display + Rand + Float + NumCast> NetworkLayer<T>
    where {
    pub fn new<GW, GB>(num_inputs: usize,
                       num_outputs: usize,
                       gen_weights: &mut GW,
                       gen_biases: &mut GB) -> NetworkLayer<T>
        where GW: FnMut(usize, usize) -> T,
              GB: FnMut(usize) -> T {
        let mut biases = Vec::with_capacity(num_outputs);
        for i in 0..num_inputs {
            biases.push(gen_biases(i));
        }
        return NetworkLayer {
            num_inputs,
            num_outputs,

            weights: Matrix::new(num_outputs, num_inputs, gen_weights),
            biases: biases,
        };
    }

    pub fn new_random<R: Rng>(num_inputs: usize, num_outputs: usize, gen: &mut R)
                              -> NetworkLayer<T> {
        let mut biases = Vec::with_capacity(num_outputs);
        let mut weights = Vec::with_capacity(num_inputs * num_outputs);
        for _i in 0..num_inputs {
            biases.push(Rand::rand(gen));
        }
        for _i in 0..num_inputs * num_outputs {
            weights.push(Rand::rand(gen));
        }
        return NetworkLayer {
            num_inputs,
            num_outputs,

            weights: Matrix::from_buffer(num_outputs, num_inputs, weights),
            biases,
        };
    }

    pub fn progagate(&self, activations: &Vec<T>, z: &mut Vec<T>, activations_next: &mut Vec<T>) {
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
            print_vector(z);
            activations_next[i] = sigmoid(z[i]);
            print_vector(activations_next);
        }
    }

    pub fn find_output_activation_error(&self,
                                        activations: &Vec<T>,
                                        z: &Vec<T>,
                                        expected: &Vec<T>,
                                        activation_errors: &mut Vec<T>) {
        //Compute the output layer error
        for k in 0..self.num_outputs {
            let d_cost_by_d_activation_k: T = T::from(2.0).unwrap()
                * (activations[k] - expected[k]);
            let d_activation_by_d_z = sigmoid_deriv(z[k]);

            activation_errors[k] = d_cost_by_d_activation_k * d_activation_by_d_z;
        }
    }

    pub fn back_propagate_layer(&self,
                                z: &Vec<T>,
                                activation_errors: &Vec<T>,
                                prev_activation_errors: &mut Vec<T>) {
        //j iterates the input layer, k iterates the output layer
        for j in 0..self.num_inputs {
            let mut total = T::zero();
            for k in 0..self.num_outputs {
                //TODO: We are calculating sigmoid_prime too many times here
                let sigmoid_prime_of_z_l_j = sigmoid_deriv(z[k]);
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
            self.layers[i].progagate(&this_activations,
                                     &mut self.z[i],
                                     &mut next_activations);
        }
    }

    pub fn back_propagate(&mut self, expected: &Vec<T>) {
        let last_layer = self.layers.len() - 1;

        self.layers[last_layer].find_output_activation_error(
            &self.activations[last_layer],
            &self.z[last_layer],
            expected,
            &mut self.activation_errors[last_layer],
        );

        self.layers[self.layers.len() - 1].find_output_activation_error(
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
    use network::sigmoid_deriv;
    use network::Network;
    use network::NetworkLayer;
    use network::sum_of_squared_error;
    use num_traits::Float;
    use rand::thread_rng;
    use rand::Rand;

    const NUMERIC_ERROR: f64 = 1e-3;

    #[test]
    fn test_sigmoid() {
        assert_approx_eq!(sigmoid( 0.0f32), 0.5f32);
        assert_approx_eq!(sigmoid( 1.0f32), 0.7310585f32);
        assert_approx_eq!(sigmoid(-1.0f32), 0.2689414f32);
    }

    #[test]
    fn test_sigmoid_deriv() {
        let x = 0.232f64;
        let dx = 0.0000001f64;

        let y = sigmoid(x);
        let dy = sigmoid(x + dx) - y;

        let numeric = dy / dx;
        let analytic = sigmoid_deriv(x);
        assert_approx_eq!(numeric, analytic, NUMERIC_ERROR);
    }

    #[test]
    fn test_layer_zeroed() {
        let layer =
            NetworkLayer::<f32>::new(1, 1,
                                     &mut |_, _| 0.0f32, &mut |_| 0.0f32);
        let a0 = vec![0.0];
        let mut z1 = vec![0.0];
        let mut a1 = vec![0.0];
        layer.progagate(&a0, &mut z1, &mut a1);
        assert_approx_eq!(z1[0], 0.0);
        assert_approx_eq!(a1[0], 0.5);
    }

    #[test]
    fn test_layer_propagate_simple() {
        let layer =
            NetworkLayer::<f32>::new(1, 1,
                                     &mut |_, _| 0.2f32, &mut |_| -0.7f32);
        let a0 = vec![0.5];
        let mut z1 = vec![0.0];
        let mut a1 = vec![0.0];
        layer.progagate(&a0, &mut z1, &mut a1);
        assert_approx_eq!(z1[0], -0.6);
        assert_approx_eq!(a1[0], 0.3543437);
    }

    #[test]
    fn test_sum_of_squared_error() {
        let a1 = vec![0.2, 0.7, 0.9];
        let c1 = sum_of_squared_error(1, &a1);

        assert_approx_eq!(c1, 0.94);
    }

    #[test]
    fn test_layer_find_activation_error() {
        let mut gen = thread_rng();
        let layer =
            NetworkLayer::<f64>::new_random(3, 2, &mut gen);

        fn compute_cost(z1: &Vec<f64>) -> f64 {
            let a1 = vec![
                sigmoid(z1[0]),
                sigmoid(z1[1])
            ];
            sum_of_squared_error(0, &a1)
        }

        let z1 = vec![
            Rand::rand(&mut gen),
            Rand::rand(&mut gen)
        ];
        let d = 0.00000001;
        let z_plus_dz1 = vec![
            z1[0] + d,
            z1[1]
        ];

        let c = compute_cost(&z1);
        let dc = compute_cost(&z_plus_dz1) - c;

        let numeric = dc / d;

        let expected = vec![1.0, 0.0];

        let mut a1_e = vec![0.0, 0.0];
        let a1 = vec![
            sigmoid(z1[0]),
            sigmoid(z1[0])
        ];
        layer.find_output_activation_error(&a1, &z1, &expected, &mut a1_e);

        assert_approx_eq!(numeric, a1_e[0], NUMERIC_ERROR);
    }

    #[test]
    fn test_layer_back_propagate_simple() {
        let mut gen = thread_rng();
        let layer =
            NetworkLayer::<f64>::new_random(3, 2, &mut gen);

        let compute_cost = |a0: &Vec<f64>| {
            let mut z1 = vec![0.0, 0.0];
            let mut a1 = vec![0.0, 0.0];
            layer.progagate(a0, &mut z1, &mut a1);
            sum_of_squared_error(0, &a1)
        };

        let a0 = vec![
            Rand::rand(&mut gen),
            Rand::rand(&mut gen),
            Rand::rand(&mut gen)
        ];
        let d= 0.00000001;

        let a0_plus_d = vec![
            a0[0] + d,
            a0[1],
            a0[2]
        ];

        let c = compute_cost(&a0);
        let dc = compute_cost(&a0_plus_d) - c;

        let numeric = dc / d;

        let expected = vec![1.0, 0.0];
        let mut a1_e = vec![0.0, 0.0];

        let mut z1 = vec![0.0, 0.0];
        let mut a1 = vec![0.0, 0.0];
        layer.progagate(&a0, &mut z1, &mut a1);

        let mut a0_e = vec![0.0, 0.0, 0.0];
        layer.find_output_activation_error(&a1, &z1, &expected, &mut a1_e);
        layer.back_propagate_layer(&z1, &a1_e, &mut a0_e);

        assert_approx_eq!(numeric, a0_e[0], NUMERIC_ERROR);
    }


    #[test]
    fn test_propagate() {
        let layer0 =
            NetworkLayer::<f32>::new(1, 1,
                                     &mut |_, _| 0.0f32, &mut |_| 0.0f32);
        let layer1 =
            NetworkLayer::<f32>::new(1, 1,
                                     &mut |_, _| 0.0f32, &mut |_| 0.0f32);
        let net = Network::new(vec![layer0, layer1]);
    }
}
