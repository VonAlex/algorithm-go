package leetcode

import "fmt"

// 正向打印 list
func listPrint(node *ListNode) {
	if node == nil {
		fmt.Println(node)
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

// 打印复杂链表
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

func getListLen(head *ListNode) int {
	lens := 0
	curr := head
	for curr != nil {
		lens++
		curr = curr.Next
	}
	return lens
}
