
extern crate byteorder;
extern crate rand;
extern crate num_traits;

mod algebra {
    pub trait Zero {
        fn zero() -> Self;
    }

    impl Zero for f32 {
        fn zero() -> f32 {
            return 0.0;
        }
    }

    impl Zero for i32 {
        fn zero() -> i32 {
            return 0;
        }
    }

    pub trait One {
        fn one() -> Self;
    }

    impl One for f32 {
        fn one() -> f32 {
            return 1.0;
        }
    }

    impl One for i32 {
        fn one() -> i32 {
            return 1;
        }
    }
}

mod real {
    use std::f32;
    use algebra::One;
    use algebra::Zero;
    use std::ops::Neg;
    use std::ops::Add;
    use std::ops::Div;
    use std::ops::Mul;

    pub trait Real
        where Self: Sized + Neg<Output=Self> + One + Zero +
        Add<Output=Self> + Mul<Output=Self> + Div<Output=Self> {
        fn exp(x: Self) -> Self;
    }

    impl Real for f32 {
        fn exp(x: f32) -> f32 {
            return f32::exp(x);
        }
    }

    pub fn sigmod<T: Real + Neg + One + Add + Div>(x: T) -> T
        where T: Neg<Output=T> + Add<Output=T> + Div<Output=T> {
        T::one() / (T::one() + Real::exp(-x))
    }
}

mod network {
    use linear::Matrix;
    use std::ops::AddAssign;
    use std::fmt::Display;
    use num_traits::{Float, NumCast};
    use rand::{Rng, Rand};

    pub fn sigmod<T: Float + NumCast>(x: T) -> T {
        let one: T = NumCast::from(1).unwrap();
        one/ (one + Float::exp(-x))
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
    }
}

mod linear {
    use num_traits::Float;
    use std::ops::Mul;
    use std::ops::Add;
    use std::ops::AddAssign;
    use std::fmt::Display;

    pub fn vec_add<T: Add + Float + Clone>(a: &Vec<T>, b: &Vec<T>) -> Vec<T>
        where
            T: Add<Output=T> {
        if a.len() != b.len() {
            panic!("cannot add vectors of different lengths");
        }

        let mut out = vec![T::zero(); a.len()];
        for i in 0..a.len() {
            out[i] = a[i].clone() + b[i].clone();
        }
        return out;
    }

    pub struct Matrix<T> {
        cols: usize,
        rows: usize,
        values: Vec<T>,
    }

    impl<T: Clone + Mul + AddAssign + Float + Display> Matrix<T>
        where
            T: Mul<Output=T> {
        pub fn new<F>(rows: usize, cols: usize, init: &mut F) -> Matrix<T>
            where F: FnMut() -> T {
            let len = rows * cols;
            let mut values = Vec::with_capacity(len);
            for _i in 0..len {
                let v = init();
                values.push(v);
            }
            Matrix { rows, cols, values }
        }

        fn set(&mut self, row: usize, col: usize, value: T) {
            self.values[col * self.rows + row] = value;
        }

        fn get(&self, row: usize, col: usize) -> T {
            self.values[col * self.rows + row].clone()
        }

        fn print(&self) {
            for row in 0..self.rows {
                for col in 0..self.cols {
                    print!("{} ", self.get(row, col));
                }
                println!();
            }
        }

        pub fn multiply_vec(&self, vec: &Vec<T>) -> Vec<T> {
            if self.cols != vec.len() {
                panic!("invalid dimentions");
            }
            let mut out = vec![T::zero(); self.rows];
            for col in 0..self.cols {
                for row in 0..self.rows {
                    out[row] += self.get(row, col) * vec[col].clone();
                }
            }
            out
        }
    }


    #[cfg(test)]
    mod tests {
        use linear::Matrix;

        #[test]
        fn matrix_vector_multiplication() {
            let mut mat = Matrix::<i32>::new(2, 3, || 0);
            mat.set(0, 0, 1);
            mat.set(0, 1, 2);
            mat.set(0, 2, 3);
            mat.set(1, 0, 4);
            mat.set(1, 1, 5);
            mat.set(1, 2, 6);

            let vec: Vec<i32> = vec![1, 2, 3];

            let v_out = mat.multiply_vec(&vec);

            assert_eq!(v_out.len(), 2);

            assert_eq!(v_out[0], 14);
            assert_eq!(v_out[1], 32);
        }
    }
}

mod data {
    use std::fs::File;
    use std::io::{Read};
    use byteorder::{BigEndian, ReadBytesExt};

    pub struct TrainingData {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<Vec<u8>>,
        pub labels: Vec<u8>
    }

    impl TrainingData {
        pub fn len(&self) -> usize {
            return self.rows * self.cols;
        }

        pub fn unpack_layer_to_f32(&self, layer_number: usize) -> Vec<f32> {
            let layer = &self.data[layer_number];
            return layer.iter().map(|v| *v as f32).collect();
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
            labels: label_out
        }
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

fn sum_of_squared_error(label: u8, output: &Vec<f32>) -> f32 {
    let label = label as usize;
    let mut total = 0.0;
    for i in 0..output.len() {
        if i == label {
            total += (output[i] - 1.0) * (output[i] - 1.0);
        } else {
            total += output[i] * output[i];
        }
    }
    return total;
}

fn main() {
    let mut rng = rand::thread_rng();
    let training_data = data::read_training_data();
    let input_to_hidden = network::NetworkLayer::<f32>::new(training_data.len(), 30, &mut rng);
    let hidden_to_output = network::NetworkLayer::<f32>::new(30, 10, &mut rng);

    for i in 0..4 {
        let this_data = training_data.unpack_layer_to_f32(i);
        let hidden = input_to_hidden.eval_layer(&this_data);
        let output = hidden_to_output.eval_layer(&hidden);
        for c in output.iter() {
            print!("{} ", c);
        }
        println!(" error = {}", sum_of_squared_error(training_data.labels[i], &output));
    }
}

