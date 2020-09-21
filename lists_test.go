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

	sum := addTwoNumbers2(l1, l2)
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

var oddEvenHead = &ListNode{
	Val: 1,
	Next: &ListNode{
		Val: 8,
		Next: &ListNode{
			Val: 3,
			Next: &ListNode{
				Val: 6,
				Next: &ListNode{
					Val: 5,
					Next: &ListNode{
						Val: 4,
						Next: &ListNode{
							Val: 7,
						},
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
	listPrint(head)
}

func TestReversePrint(t *testing.T) {
	listPrint(head)
	fmt.Println(reversePrint4(head))
	listPrint(oneNodeHead)
	fmt.Println(reversePrint4(oneNodeHead))
}

func TestReverseList(t *testing.T) {
	listPrint(head)
	listPrint(reverseList(head))
}

func TestDeleteNode(t *testing.T) {
	listPrint(head3)
	listPrint(deleteNode(head3, 2))
}

func TestRemoveElements(t *testing.T) {
	listPrint(head3)
	// 2->1->1->2
	// 1->1
	// 1
	listPrint(removeElements(head3, 1))
}

func TestPrintCommonPart(t *testing.T) {
	printCommonPart(head, head2)
}

func TestRemoveLastKthNode(t *testing.T) {
	listPrint(head)
	listPrint(removeLastKthNode4(head, 1))
}

func TestRemoveDuplicateNodes(t *testing.T) {
	listPrint(head2)
	removeDuplicateNodes(head2)
	listPrint(head2)
}

func TestDetectCycle(t *testing.T) {
	listPrint(detectCycle(oneNodeHead))
}

func TestMergeTwoLists(t *testing.T) {
	l1 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 3,
			Next: &ListNode{
				Val: 9,
				Next: &ListNode{
					Val: 10,
				},
			},
		},
	}
	l2 := &ListNode{
		Val: 2,
		Next: &ListNode{
			Val: 4,
			Next: &ListNode{
				Val: 5,
				Next: &ListNode{
					Val: 7,
				},
			},
		},
	}
	listPrint(mergeTwoLists3(l1, l2))
}

func TestDeleteMiddleNode(t *testing.T) {
	listPrint(head)
	listPrint(deleteMiddleNode2(head))
}

func TestIsPalindromeList(t *testing.T) {
	listPrint(head3)
	t.Log(isPalindromeList(head3))
}

func TestReverseKGroup(t *testing.T) {
	listPrint(head)
	listPrint(reverseKGroup2(head, 3))
}

func TestSortList(t *testing.T) {
	// listPrint(head)
	listPrint(sortList3(head3))
}

func TestReverseN(t *testing.T) {
	listPrint(head3)
	listPrint(reverseN(head3, 4))
}
func TestOddEvenList(t *testing.T) {
	listPrint(head3)
	listPrint(oddEvenList(head3))
}

func TestDeleteDuplicates(t *testing.T) {
	listPrint(head3)
	listPrint(deleteDuplicates(head3))
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
	copyList := copyRandomList(head)
	printComplexList(head)
	printComplexList(copyList)
}

func TestDivideOddEvenList(t *testing.T) {
	listPrint(oddEvenHead)
	listPrint(oddEvenSortlist(oddEvenHead))
}

func TestRotateRight(t *testing.T) {
	listPrint(oddEvenHead)
	listPrint(rotateRight2(oddEvenHead, 3))
}

func Test_reorderList(t *testing.T) {
	// 1->2->3->4->5
	listPrint(head)
	reorderList(head)
	// 1->5->2->4->3
	listPrint(head)

	// 1->2->2->3
	listPrint(head3)
	reorderList(head3)
	// 1->3->2->2
	listPrint(head3)
}
