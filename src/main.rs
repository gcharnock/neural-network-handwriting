struct Network {
    layers: Vec<Vec<f32>>
}

fn main() {
    println!("Hello, world!");
}

mod zero {
    pub trait Zero {
        fn zero() -> Self;
    }

    impl Zero for f32 {
        fn zero() -> f32 {
            return 0.0;
        }
    }
}

mod matrix {
    use zero::Zero;
    use std::ops::Mul;
    use std::ops::AddAssign;
    use std::ops::IndexMut;

    pub struct Matrix<T> {
        width: usize,
        height: usize,
        values: Vec<T>,
    }

    impl<T: Clone + Mul + AddAssign + Zero> Matrix<T>
        where
            T: Mul<Output=T> {
        fn new(width: usize, height: usize) -> Matrix<T> {
            return Matrix {
                width,
                height,
                values: vec![T::zero(); width * height],
            };
        }

        fn set(&mut self, row: usize, column: usize, value: T) {
            self.values[column * self.width + row] = value;
        }

        fn get(&self, row: usize, column: usize) -> T {
            self.values[column * self.width + row].clone()
        }

        fn multiply_vec(&self, vec: &Vec<T>) -> Vec<T> {
            if self.width != vec.len() {
                panic!("invalid dimentions");
            }
            let mut out = vec![T::zero(); vec.len()];
            for i in 0..self.height {
                for j in 0..self.width {
                    out[j] += self.get(i, j) * vec[j].clone();
                }
            }
            out
        }
    }


    #[cfg(test)]
    mod tests {
        use matrix::Matrix;

        #[test]
        fn matrix_multiplication() {
            let mat = Matrix::<f32>::new(3, 3);
        }
    }
}


