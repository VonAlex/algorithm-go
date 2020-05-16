package leetcode

import "testing"

func TestIsValid(t *testing.T) {
	s := "()"
	s = "()[]{}"
	s = "(]"
	s = "([)]"
	s = "{[]}"
	t.Log(IsValid(s))
}
