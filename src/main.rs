fn main() {
    println!("Hello, world!");
}

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
    use std::ops;
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
    use std::ops::Mul;
    use std::ops::AddAssign;
    use std::ops::IndexMut;
    use std::fmt::Display;
    use algebra::Zero;
    use algebra::One;
    use real::sigmod;
    use real::Real;

    struct NetworkLayer<T> {
        num_inputs: usize,
        num_outputs: usize,

        weights: Matrix<T>,
        biases: Vec<T>,
    }

    impl<T: Clone + AddAssign + Display> NetworkLayer<T>
        where
            T: Real {
        fn new(num_inputs: usize, num_outputs: usize) -> NetworkLayer<T> {
            return NetworkLayer {
                num_inputs,
                num_outputs,

                weights: Matrix::new(num_outputs, num_inputs),
                biases: vec![T::zero(); num_inputs],
            };
        }

        fn eval_layer(&self, inputs: &Vec<T>) -> Vec<T> {
            if self.num_inputs != inputs.len() {
                panic!("incorrect number of inputs to layer");
            }

            let propagated = self.weights.multiply_vec(inputs);
            return propagated.into_iter().map(sigmod).collect();
        }
    }
}

mod linear {
    use algebra::Zero;
    use std::ops::Mul;
    use std::ops::Add;
    use std::ops::AddAssign;
    use std::ops::IndexMut;
    use std::fmt::Display;

    pub fn vec_add<T: Add + Zero + Clone>(a: &Vec<T>, b: &Vec<T>) -> Vec<T>
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

    impl<T: Clone + Mul + AddAssign + Zero + Display> Matrix<T>
        where
            T: Mul<Output=T> {
        pub fn new(rows: usize, cols: usize) -> Matrix<T> {
            return Matrix {
                rows,
                cols,
                values: vec![T::zero(); rows * cols],
            };
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
            let mut mat = Matrix::<i32>::new(2, 3);
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


