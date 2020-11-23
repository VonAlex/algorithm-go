package leetcode

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestTwo3Sum(t *testing.T) {
	type test struct {
		input  []int
		target int
		want   []int
	}

	tests := []test{
		{input: []int{2, 7, 11, 15}, target: 9, want: []int{0, 1}},
		{input: []int{2, 7, 11, 15}, target: 20, want: []int{0, 0}},
		{input: []int{3, 2, 4}, target: 6, want: []int{1, 2}},
	}

	for _, tc := range tests {
		got := TwoSum2(tc.input, tc.target)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestFindRepeatNumber(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{input: []int{2, 3, 1, 0, 2, 5, 3}, want: 2},
		{input: []int{2, 3, 1, 0}, want: -1},
	}
	for _, tc := range tests {
		got := FindRepeatNumber2(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestDistributeCandies(t *testing.T) {
	type test struct {
		candies int
		people  int
		want    []int
	}
	tests := []test{
		{candies: 7, people: 4, want: []int{1, 2, 3, 1}},
		{candies: 10, people: 3, want: []int{5, 2, 3}},
	}
	for _, tc := range tests {
		got := DistributeCandies(tc.candies, tc.people)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestExange(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5}
	t.Log(Exchange(nums))
}

func TestPivotIndex(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{1, 7, 3, 6, 5, 6}, 3},
		{[]int{1, 2, 3}, -1},
		{[]int{}, -1},
	}
	for _, tc := range tests {
		got := pivotIndex(tc.input)
		if got != tc.want {
			t.Log(tc.want, got)
		}
	}
}

func TestDominantIndex(t *testing.T) {
	nums := []int{0, 0, 0, 1}
	t.Log(DominantIndex2(nums))
}

func TestSingleNumber(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{4, 1, 2, 1, 2}, 4},
		{[]int{2, 2, 1}, 1},
	}
	for _, tc := range tests {
		got := SingleNumber(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestSingleNumber2(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{2, 2, 3, 2}, 3},
	}
	for _, tc := range tests {
		got := SingleNumber22(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestMajorityElement(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{1}, 1},
		{[]int{3, 3, 1}, 3},
		{[]int{2, 2, 1, 1, 1, 2, 2}, 2},
	}
	for _, tc := range tests {
		got := MajorityElement3(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestTwoSum(t *testing.T) {
	numbers := []int{3, 24, 50, 79, 88, 150, 345}
	target := 200
	t.Log(TwoSum5(numbers, target))
}

func TestCountPrimes(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{5}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CountPrimes(tt.args.n); got != tt.want {
				t.Errorf("CountPrimes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSubarraySum(t *testing.T) {
	type args struct {
		nums []int
		k    int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 1}, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := subarraySum4(tt.args.nums, tt.args.k); got != tt.want {
				t.Errorf("SubarraySum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMoveZeroes(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
	}{
		{"normal", args{[]int{0, 1, 0, 3, 12}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			MoveZeroes2(tt.args.nums)
		})
	}
}

func TestRemoveElement(t *testing.T) {
	type args struct {
		nums []int
		val  int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{3, 2, 2, 3}, 3}, 2},
		{"normal2", args{[]int{0, 1, 2, 2, 3, 0, 4, 2}, 2}, 5},
		{"normal2", args{[]int{1, 2, 3, 5, 4}, 5}, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveElement(tt.args.nums, tt.args.val); got != tt.want {
				t.Errorf("RemoveElement() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoveDuplicates(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 2}}, 2},
		{"normal2", args{[]int{0, 0, 1, 1, 1, 2, 2, 3, 3, 4}}, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveDuplicates(tt.args.nums); got != tt.want {
				t.Errorf("RemoveDuplicates() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoveDuplicates2(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 1, 2, 2, 3}}, 5},
		{"normal", args{[]int{0, 0, 1, 1, 1, 1, 2, 3, 3}}, 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveDuplicates2(tt.args.nums); got != tt.want {
				t.Errorf("RemoveDuplicates2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMerge(t *testing.T) {
	num1 := make([]int, 6)
	copy(num1, []int{4, 5, 6})
	m := 3
	num2 := []int{1, 2, 3}
	n := 3
	Merge(num1, m, num2, n)
	fmt.Println(num1)
}

func TestFindKthLargest(t *testing.T) {
	nums := []int{3, 2, 1, 5, 6, 4}
	k := 2
	// nums := []int{1}
	// k := 1

	nums = []int{3, 2, 3, 1, 2, 4, 5, 5, 6}
	k = 4
	t.Log(findKthLargest(nums, k))
}

func TestMaxSubArray(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"n1", args{[]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}}, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxSubArray3(tt.args.nums); got != tt.want {
				t.Errorf("MaxSubArray() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRotate(t *testing.T) {
	nums := []int{1, 2}
	k := 3
	rotate(nums, k)
	t.Log(nums)
}

func Test_intersect(t *testing.T) {
	nums1 := []int{1, 2, 2, 1}
	nums2 := []int{2, 2}
	// nums1 := []int{4, 9, 5}
	// nums2 := []int{9, 4, 9, 8, 4}
	t.Log(intersect2(nums1, nums2))
}

func Test_plusOne(t *testing.T) {
	type args struct {
		digits []int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"case1", args{[]int{9}}, []int{1, 0}},
		{"case2", args{[]int{1, 2, 3}}, []int{1, 2, 4}},
		{"case3", args{[]int{1, 2, 9}}, []int{1, 3, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := plusOne(tt.args.digits); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("plusOne() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_rotateMatrix(t *testing.T) {
	matrix1 := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	rotateMatrix(matrix1)
	t.Log(matrix1)

	matrix2 := [][]int{
		{5, 1, 9, 11},
		{2, 4, 8, 10},
		{13, 3, 6, 7},
		{15, 14, 12, 16},
	}
	rotateMatrix(matrix2)
	t.Log(matrix2)
}

func Test_mergeArr(t *testing.T) {
	nums1 := []int{1, 2, 3, 0, 0, 0}
	nums2 := []int{2, 5, 6}
	mergeArr(nums1, 3, nums2, 3)
	t.Log(nums1)
}

func Test_maxProfit(t *testing.T) {
	// prices := []int{7, 1, 5, 3, 6, 4}
	prices := []int{7, 1, 5, 3, 6, 2}
	t.Log(maxProfit3(prices))
}

func Test_SumRange(t *testing.T) {
	nums := []int{-2, 0, 3, -5, 2, -1}
	sumArr := Constructor2(nums)
	t.Log(sumArr.SumRange(0, 2) == 1)
	t.Log(sumArr.SumRange(2, 5) == -1)
	t.Log(sumArr.SumRange(0, 5) == -3)
}

// func Test_findMedianSortedArrays(t *testing.T) {
// 	// num1 := []int{0, 0}
// 	// num2 := []int{0, 0}
// 	// num1 := []int{1, 3}
// 	// num2 := []int{2}
// 	t.Log(findMedianSortedArrays3(num1, num2))
// }

func Test_findMedianSortedArrays(t *testing.T) {
	type args struct {
		nums1 []int
		nums2 []int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{"case0", args{[]int{0, 0}, []int{0, 0}}, 0},
		{"case1", args{[]int{1}, []int{2}}, 1.5},
		{"case2", args{[]int{1, 3}, []int{2}}, 2},
		{"case3", args{[]int{5}, []int{1, 2, 3}}, 2.5},
		{"case4", args{[]int{1, 2}, []int{3, 4}}, 2.5},
		{"case5", args{[]int{}, []int{}}, 0},
		{"case6", args{[]int{2}, []int{}}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findMedianSortedArrays3(tt.args.nums1, tt.args.nums2); got != tt.want {
				t.Errorf("findMedianSortedArrays() = %v, want %v", got, tt.want)
			}
		})
	}
}
