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
		Val: 3,
		Next: &ListNode{
			Val: 5,
			Next: &ListNode{
				Val: 4,
				Next: &ListNode{
					Val: 2,
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

var head3 = &ListNode{
	Val: 1,
	Next: &ListNode{
		Val: 2,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
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
	ListPrint(ReverseList(head))
}

func TestDeleteNode(t *testing.T) {
	ListPrint(head3)
	ListPrint(DeleteNode(head3, 2))
}

func TestRemoveElements(t *testing.T) {
	ListPrint(head3)
	// 2->1->1->2
	// 1->1
	// 1
	ListPrint(RemoveElements(head3, 1))
}

func TestPrintCommonPart(t *testing.T) {
	PrintCommonPart(head, head2)
}

func TestRemoveLastKthNode(t *testing.T) {
	ListPrint(head)
	ListPrint(RemoveLastKthNode4(head, 1))
}

func TestRemoveDuplicateNodes(t *testing.T) {
	ListPrint(head2)
	RemoveDuplicateNodes(head2)
	ListPrint(head2)
}

func TestDetectCycle(t *testing.T) {
	ListPrint(DetectCycle(oneNodeHead))
}

func TestMergeTwoLists(t *testing.T) {
	l1 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
			},
		},
	}
	l2 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 3,
			Next: &ListNode{
				Val: 4,
			},
		},
	}
	ListPrint(MergeTwoLists(l1, l2))
}

func TestDeleteMiddleNode(t *testing.T) {
	ListPrint(head)
	ListPrint(DeleteMiddleNode2(head))
}

func TestIsPalindromeList(t *testing.T) {
	ListPrint(head3)
	t.Log(IsPalindromeList(head3))
}

func TestReverseKGroup(t *testing.T) {
	ListPrint(head)
	ListPrint(ReverseKGroup2(head, 2))
}

func TestSortList(t *testing.T) {
	ListPrint(head3)
	ListPrint(SortList(head3))
}

func TestReverseN(t *testing.T) {
	ListPrint(head3)
	ListPrint(ReverseN(head3, 4))
}
func TestOddEvenList(t *testing.T) {
	ListPrint(head3)
	ListPrint(OddEvenList(head3))
}

func TestDeleteDuplicates(t *testing.T) {
	ListPrint(head3)
	ListPrint(DeleteDuplicates(head3))
}

func TestCopyRandomList(t *testing.T) {
	head := &ComplexNode{
		Val: 7,
	}
	node1 := &ComplexNode{
		Val:    13,
		Random: head,
	}
	node4 := &ComplexNode{
		Val:    1,
		Random: head,
	}
	node2 := &ComplexNode{
		Val:    11,
		Random: node4,
	}
	node3 := &ComplexNode{
		Val:    10,
		Random: node2,
	}
	head.Next = node1
	node1.Next = node2
	node2.Next = node3
	node3.Next = node4

	printComplexList(head)
	copyList := CopyRandomList(head)
	printComplexList(head)
	printComplexList(copyList)
}

func printComplexList(node *ComplexNode) {
	curr := node
	nodes := make(map[int]int)
	idx := 0
	for curr != nil {
		nodes[curr.Val] = idx
		idx++
		curr = curr.Next
	}
	curr = node
	var res [][]interface{}
	for curr != nil {
		if curr.Random != nil {
			res = append(res, []interface{}{curr.Val, nodes[curr.Random.Val]})
		} else {
			res = append(res, []interface{}{curr.Val, nil})
		}
		curr = curr.Next
	}
	fmt.Println(res)
}
