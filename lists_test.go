package leetcode

import (
	"testing"
)

func Test_AddTwoNumbers(t *testing.T) {
	l1 := &ListNode{
		Val: 7,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 4,
				Next: &ListNode{
					Val: 3,
				},
			},
		},
	}

	l2 := &ListNode{
		Val: 5,
		Next: &ListNode{
			Val: 6,
			Next: &ListNode{
				Val: 4,
			},
		},
	}

	listPrint(addTwoNumbers4(l1, l2))
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
	// listPrint(head)
	// fmt.Println(reversePrint4(head))
	// listPrint(oneNodeHead)
	// fmt.Println(reversePrint4(oneNodeHead))

	t.Log(reversePrint5(head))
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
	listPrint(removeLastKthNode(head, 1))
}

func Test_removeNthFromEnd(t *testing.T) {
	listPrint(head)
	listPrint(removeNthFromEnd(head, 1))
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
			// Next: &ListNode{
			// 	Val: 9,
			// 	Next: &ListNode{
			// 		Val: 10,
			// 	},
			// },
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
	l4 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
				Next: &ListNode{
					Val: 4,
				},
			},
		},
	}
	listPrint(deleteMiddleNode(l4))
	l1 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
			},
		},
	}
	listPrint(deleteMiddleNode(l1))
	l2 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
		},
	}
	listPrint(deleteMiddleNode(l2))
	l3 := &ListNode{
		Val: 1,
	}
	listPrint(deleteMiddleNode(l3))
	listPrint(deleteMiddleNode(nil))
}

func TestIsPalindromeList(t *testing.T) {
	listPrint(head3)
	t.Log(isPalindromeList3(head3))
}

func TestReverseKGroup(t *testing.T) {
	listPrint(head)
	listPrint(reverseKGroup6(head, 2))
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
	listPrint(head)
	listPrint(oddEvenList(head))
}

func TestDeleteDuplicates(t *testing.T) {
	listPrint(head3)
	listPrint(deleteDuplicates3(head3))
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
	listPrint(head)
	listPrint(rotateRight(head, 3))
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

func Test_getListIntersection(t *testing.T) {
	common := &ListNode{
		Val: 7,
		Next: &ListNode{
			Val: 9,
			Next: &ListNode{
				Val: 10,
			},
		},
	}
	l1 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val:  2,
			Next: common,
		},
	}
	l2 := &ListNode{
		Val:  5,
		Next: common,
	}
	// 都无环，相交
	n := getListIntersection(l1, l2)
	if n == nil {
		t.Log(n)
	} else {
		t.Log(n.Val)
	}

	l3 := &ListNode{
		Val: 5,
		Next: &ListNode{
			Val: 6,
			Next: &ListNode{
				Val: 10,
			},
		},
	}
	// 都无环，不相交
	n = getListIntersection(l1, l3)
	if n == nil {
		t.Log(n)
	} else {
		t.Log(n.Val)
	}

	n1 := &ListNode{Val: 1}
	n2 := &ListNode{Val: 2}
	n3 := &ListNode{Val: 3}
	n4 := &ListNode{Val: 4}
	n5 := &ListNode{Val: 5}
	n6 := &ListNode{Val: 6}
	n7 := &ListNode{Val: 7}
	n8 := &ListNode{Val: 8}

	n1.Next = n2
	n2.Next = n3
	n3.Next = n4
	n4.Next = n5
	n5.Next = n3

	n6.Next = n7
	n7.Next = n8
	n8.Next = n4

	n = getListIntersection(n1, n6)
	if n == nil {
		t.Log(n)
	} else {
		t.Log(n.Val)
	}
}

func Test_reverseBetween(t *testing.T) {
	listPrint(head)
	listPrint(reverseBetween(head, 2, 4))
}

func Test_swapPairs(t *testing.T) {
	listPrint(head)
	listPrint(swapPairs(head))
}

func Test_listPlusOne(t *testing.T) {
	h1 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
			},
		},
	}
	listPrint(h1)
	listPrint(listPlusOne2(h1))
}

func Test_splitListToParts(t *testing.T) {
	h2 := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 2,
			Next: &ListNode{
				Val: 3,
			},
		},
	}
	arr := splitListToParts(h2, 5)
	for _, l := range arr {
		listPrint(l)
	}

	arr = splitListToParts(nil, 3)
	for _, l := range arr {
		listPrint(l)
	}
}

func Test_partition2(t *testing.T) {
	h := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 4,
			Next: &ListNode{
				Val: 3,
				Next: &ListNode{
					Val: 2,
					Next: &ListNode{
						Val: 5,
						Next: &ListNode{
							Val: 2,
						},
					},
				},
			},
		},
	}
	listPrint(partitionList2(h, 3))
}

func Test_getDecimalValue(t *testing.T) {
	h := &ListNode{
		Val: 1,
		Next: &ListNode{
			Val: 0,
			Next: &ListNode{
				Val: 1,
			},
		},
	}
	t.Log(getDecimalValue(h))
}
