#[cfg_attr(test, macro_use)]
extern crate assert_approx_eq;

extern crate byteorder;
extern crate rand;
extern crate num_traits;

mod real {
    use std::f32;
    use std::ops::Neg;
    use std::ops::Add;
    use std::ops::Div;
    use std::ops::Mul;
    use num_traits::Float;

    pub trait Real
        where Self: Sized + Neg<Output=Self> +
        Add<Output=Self> + Mul<Output=Self> + Div<Output=Self> {
        fn exp(x: Self) -> Self;
    }

    impl Real for f32 {
        fn exp(x: f32) -> f32 {
            return f32::exp(x);
        }
    }

    pub fn sigmoid<T: Real + Float + Neg + Add + Div>(x: T) -> T
        where T: Neg<Output=T> + Add<Output=T> + Div<Output=T> {
        T::one() / (T::one() + Real::exp(-x))
    }

    pub fn sigmoid_prime<T: Neg + Add + Div + Float>(x: T) -> T
        where T: Neg<Output=T> + Add<Output=T> + Div<Output=T> {
        let exp_neg_x = T::exp(-x);
        let one_minus_exp_neg_x = T::one() - exp_neg_x;
        return -exp_neg_x / one_minus_exp_neg_x / one_minus_exp_neg_x;
    }
}

mod network;
mod linear;

mod data {
    use std::fs::File;
    use std::io::Read;
    use byteorder::{BigEndian, ReadBytesExt};

    pub struct TrainingData {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<Vec<u8>>,
        pub labels: Vec<u8>,
    }

    impl TrainingData {
        pub fn len(&self) -> usize {
            return self.rows * self.cols;
        }

        pub fn unpack_layer_to_f32(&self, layer_number: usize) -> Vec<f32> {
            let layer = &self.data[layer_number];
            return layer.iter().map(|v| (*v as f32 / 255.0)).collect();
        }
    }

    pub fn read_training_data() -> TrainingData {
        let mut label_file = File::open("data/train-labels-idx1-ubyte").unwrap();
        let label_magic_number = label_file.read_i32::<BigEndian>().unwrap();
        let label_item_count = label_file.read_i32::<BigEndian>().unwrap() as usize;

        let mut label_out = Vec::<u8>::with_capacity(label_item_count);

        if label_magic_number != 2049 {
            panic!("did got get expected magic number from file. Got {}", label_magic_number);
        }

        let mut file = File::open("data/train-images-idx3-ubyte").unwrap();
        let magic_number = file.read_i32::<BigEndian>().unwrap();
        let item_count = file.read_i32::<BigEndian>().unwrap() as usize;
        let rows = file.read_i32::<BigEndian>().unwrap() as usize;
        let cols = file.read_i32::<BigEndian>().unwrap() as usize;
        let pixels = rows * cols;
        let mut out = Vec::<Vec<u8>>::with_capacity(item_count);

        if magic_number != 2051 {
            panic!("did got get expected magic number from file");
        }

        if label_item_count != item_count {
            panic!("item counts did not match between data and label file");
        }

        println!("item count = {}, rows = {}, cols = {}", item_count, rows, cols);
        for i in 0..item_count {
            let mut data = Vec::<u8>::with_capacity(pixels);
            unsafe { data.set_len(pixels); }
            let bytes_read = file.read(&mut data[..]).unwrap();
            if bytes_read != pixels {
                panic!("Unable to read full image (read {}, expected {})", bytes_read, pixels);
            }

            let label = label_file.read_u8().unwrap();
            if i < 3 {
                println!("should be {}", label);
                print_image(cols, &data);
            }
            out.push(data);
            label_out.push(label);
        }
        return TrainingData {
            rows,
            cols,
            data: out,
            labels: label_out,
        };
    }

    pub fn print_image(cols: usize, image: &Vec<u8>) {
        let rows = image.len() / cols;
        for row in 0..rows {
            for col in 0..cols {
                let v = image[cols * row + col];
                if v > 240 {
                    print!("#");
                } else if v > 220 {
                    print!("*");
                } else if v > 128 {
                    print!(".");
                } else {
                    print!(" ");
                }
            }
            println!();
        }
    }
}

use std::fmt::Display;
use network::{Network, NetworkLayer};
use linear::print_vector;
use rand::Rand;
use num_traits::NumCast;


fn make_expected(label: u8) -> Vec<f32> {
    let mut expected = Vec::with_capacity(10);
    for i in 0..10 {
        if label == i {
            expected.push(1.0);
        } else {
            expected.push(0.0);
        }
    }
    return expected;
}


fn main() {
    let mut gen_w = rand::thread_rng();
    let mut gen_b = rand::thread_rng();

    let mut gen_weights = |_, _| {
        let r: f32 = Rand::rand(&mut gen_w);
        r - 0.5f32
    };

    let mut gen_biases = |_| {
        let r: f32 = Rand::rand(&mut gen_b);
        r - 0.5f32
    };

    let training_data = data::read_training_data();
    let input_to_hidden = NetworkLayer::<f32>::new(
        training_data.len(), 30,
        &mut gen_weights, &mut gen_biases);

    let hidden_to_output = NetworkLayer::<f32>::new(
        30, 10,
        &mut gen_weights, &mut gen_biases);

    let layers = vec![input_to_hidden, hidden_to_output];
    let mut network = Network::new(layers);

    network.describe();

    for i in 0..1 {
        let this_data = training_data.unpack_layer_to_f32(i);
        let expected = make_expected(training_data.labels[i]);

        network.set_input_activations(&this_data);
        network.propagate();

        for activations in network.activations.iter() {
            print!("activations = ");
            print_vector(&activations);
            println!();
        }

        network.back_propagate(&expected);

        for errors in network.activation_errors.iter() {
            print!("activation_error = ");
            print_vector(&errors);
            println!();
        }
    }
}

