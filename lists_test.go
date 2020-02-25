package leetcode

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestAddTwoNumbers(t *testing.T) {
	l1 := &ListNode{
		Val: 5,
		// Next: &ListNode{
		// 	Val: 4,
		// 	Next: &ListNode{
		// 		Val: 3,
		// 	},
		// },
	}

	l2 := &ListNode{
		Val: 5,
		// Next: &ListNode{
		// 	Val: 6,
		// 	Next: &ListNode{
		// 		Val: 4,
		// 	},
		// },
	}

	sum := AddTwoNumbers2(l1, l2)
	s, _ := json.Marshal(sum)
	t.Log(string(s))
}

var head = &ListNode{
	Val: 1,
	Next: &ListNode{
		Val: 2,
		Next: &ListNode{
			Val: 3,
			Next: &ListNode{
				Val: 4,
				Next: &ListNode{
					Val: 5,
				},
			},
		},
	},
}

var oneNodeHead = &ListNode{
	Val: 1,
}

func TestListPrint(t *testing.T) {
	ListPrint(head)
}

func TestReversePrint(t *testing.T) {
	ListPrint(head)
	fmt.Println(ReversePrint4(head))
	ListPrint(oneNodeHead)
	fmt.Println(ReversePrint4(oneNodeHead))
}
