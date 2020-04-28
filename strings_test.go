package leetcode

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

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

func TestReverse(t *testing.T) {
	type test struct {
		input int
		want  int
	}
	tests := []test{
		{input: 1234, want: 4321},
		{input: -1234, want: -4321},
		{input: 120, want: 21},
	}
	for _, tc := range tests {
		got := Reverse2(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestIsPalindrome(t *testing.T) {
	type test struct {
		input string
		want  bool
	}
	tests := []test{
		{input: "A man, a plan, a canal: Panama", want: true},
		{input: "race a car", want: false},
	}
	for _, tc := range tests {
		got := IsPalindrome(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestPalindromePartition(t *testing.T) {
	s := "aab"
	t.Log(PalindromePartition3(s))
}

func TestNumSum(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"no num", args{"cccccc"}, 0},
		{"all posotive", args{"a1cd2e33"}, 36},
		{"one negative sign", args{"a1cd-2e33"}, 32},
		{"two negative sign", args{"a1cd--2e33"}, 36},
		{"head negative", args{"-1a1c-d--2e33"}, 35},
		{"head negative", args{"a1c-d--2e-33"}, -30},
		{"other", args{"a1c-d--2e33"}, 36},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NumSum(tt.args.s); got != tt.want {
				t.Errorf("NumSum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSundaySearch(t *testing.T) {
	tt := "aaaaa"
	p := "bba"
	t.Log(SundaySearch(tt, p))
}

func TestReverseWords(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{"have left/right space", args{"  hello world!  "}, "world! hello"},
		{"have middle space", args{"a good   example"}, "example good a"},
		{"no space", args{"a good example"}, "example good a"},
		{"empty", args{""}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReverseWords3(tt.args.s); got != tt.want {
				t.Errorf("ReverseWords() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCountFields(t *testing.T) {
	s := "     "
	t.Log(CountFields(s))
}

func TestSplitWords(t *testing.T) {
	s := "a good   example"
	t.Log(SplitWords(s))
}

func TestMinWindow(t *testing.T) {
	type args struct {
		s string
		t string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{"normal", args{"ADOBECODEBANC", "ABC"}, "BANC"},
		{"normal2", args{"ABAACBAB", "ABC"}, "ACB"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinWindow(tt.args.s, tt.args.t); got != tt.want {
				t.Errorf("MinWindow() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFindAnagrams(t *testing.T) {
	type args struct {
		s string
		p string
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"normal", args{"cbaebabacd", "abc"}, []int{0, 6}},
		{"normal2", args{"abab", "ab"}, []int{0, 1, 2}},
		{"normal3", args{"baa", "aa"}, []int{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FindAnagrams(tt.args.s, tt.args.p); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FindAnagrams() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLengthOfLongestSubstring(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{"abcabcbb"}, 3},
		{"normal2", args{"bbbbb"}, 1},
		{"normal3", args{"pwwkew"}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LengthOfLongestSubstring(tt.args.s); got != tt.want {
				t.Errorf("LengthOfLongestSubstring() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestReverseVowels(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want string
	}{
		{"contain upper", args{"aA"}, "Aa"},
		{"contain punch", args{"a.b,."}, "a.b,."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ReverseVowels2(tt.args.s); got != tt.want {
				t.Errorf("ReverseVowels() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxArea(t *testing.T) {
	type args struct {
		height []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"contain upper", args{[]int{1, 8, 6, 2, 5, 4, 8, 3, 7}}, 49},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxArea(tt.args.height); got != tt.want {
				t.Errorf("MaxArea() = %v, want %v", got, tt.want)
			}
		})
	}
}
