package leetcode

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestLengthOfLongestSubstring(t *testing.T) {
	type test struct {
		input string
		want  int
	}
	tests := []test{
		{input: "abcabcbb", want: 3},
		{input: "bbbbb", want: 1},
		{input: "pwwkew", want: 3},
	}

	for _, tc := range tests {
		got := LengthOfLongestSubstring(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestReplaceSpace(t *testing.T) {
	type test struct {
		input string
		want  string
	}
	tests := []test{
		{input: "we are happy", want: "we%20are%20happy"},
		{input: "wearehappy", want: "wearehappy"},
		{input: "we  are happy", want: "we%20%20are%20happy"},
		{input: " we are happy", want: "%20we%20are%20happy"},
	}

	for _, tc := range tests {
		got := ReplaceSpace(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestFirstUniqChar(t *testing.T) {
	t.Log(FirstUniqChar("leetcode"))
}
