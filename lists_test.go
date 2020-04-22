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

var head2 = &ListNode{
	Val: 1,
	Next: &ListNode{
		Val: 2,
		Next: &ListNode{
			Val: 3,
			Next: &ListNode{
				Val: 3,
				Next: &ListNode{
					Val: 2,
					Next: &ListNode{
						Val: 1,
					},
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

func TestReverseList(t *testing.T) {
	ListPrint(head)
	ListPrint(ReverseList2(head))
}

func TestDeleteNode(t *testing.T) {
	ListPrint(head)
	ListPrint(DeleteNode(head, 2))
}

func TestPrintCommonPart(t *testing.T) {
	PrintCommonPart(head, head2)
}

func TestRemoveLastKthNode(t *testing.T) {
	ListPrint(head)
	ListPrint(RemoveLastKthNode3(head, 5))
}

func TestRemoveDuplicateNodes(t *testing.T) {
	ListPrint(head2)
	RemoveDuplicateNodes(head2)
	ListPrint(head2)
}

func TestDetectCycle(t *testing.T) {
	ListPrint(DetectCycle(oneNodeHead))
}
