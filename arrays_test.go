package leetcode

import (
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
