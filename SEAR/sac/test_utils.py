import jax.numpy as np
from numpy.testing import assert_array_equal

from sac.utils import roll_rows


class TestRollPerRowVectorized:
    def test_basic_functionality(self):
        """Test basic functionality with simple array and shifts."""
        arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        shifts = np.array([1, -2, 3])
        expected = np.array(
            [
                [5, 1, 2, 3, 4],  # Shift right by 1
                [8, 9, 10, 6, 7],  # Shift left by 2
                [13, 14, 15, 11, 12],  # Shift right by 3
            ]
        )

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, expected)

    def test_zero_shifts(self):
        """Test with zero shifts - array should remain unchanged."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        shifts = np.array([0, 0])

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, arr)

    def test_all_same_shift(self):
        """Test with the same shift for all rows - should be equivalent to np.roll on axis 1."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        shifts = np.array([2, 2])
        expected = np.roll(arr * 1.0, 2, axis=1)

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, expected)

    def test_larger_than_width_shifts(self):
        """Test shifts larger than the width of the array."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        shifts = np.array([4, -4])  # Equivalent to shifts of 1 and -1 for a width of 3
        expected = np.array([[3, 1, 2], [5, 6, 4]])

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, expected)

    def test_single_row(self):
        """Test with a single row array."""
        arr = np.array([[1, 2, 3, 4, 5]])
        shifts = np.array([2])
        expected = np.array([[4, 5, 1, 2, 3]])

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, expected)

    def test_single_column(self):
        """Test with a single column array."""
        arr = np.array([[1], [2], [3]])
        shifts = np.array([0, 0, 0])  # Any shift doesn't change a single column

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, arr)

    def test_dtype_preservation(self):
        """Test that the function preserves the dtype of the input array."""
        arr = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
        shifts = np.array([1, 2])

        result = roll_rows(arr * 1.0, shifts)
        assert result.dtype == arr.dtype

    def test_empty_array(self):
        """Test with an empty array (0 rows, some columns)."""
        arr = np.empty((0, 5))
        shifts = np.array([], dtype=int)

        result = roll_rows(arr * 1.0, shifts)
        assert result.shape == arr.shape
        assert len(result) == 0

    def test_negative_shifts(self):
        """Test with negative shifts (left rolls)."""
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        shifts = np.array([-1, -2])
        expected = np.array([[2, 3, 4, 1], [7, 8, 5, 6]])

        result = roll_rows(arr * 1.0, shifts)
        assert_array_equal(result, expected)
