package leetcode

import "fmt"

// ListPrint 正向打印 list
func listPrint(node *ListNode) {
	if node == nil {
		return
	}
	for node != nil {
		fmt.Print(node.Val)
		if node.Next != nil {
			fmt.Print("->")
		}
		node = node.Next
	}
	fmt.Println()
}
