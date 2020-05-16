package leetcode

import (
	"container/list"
)

/**
 * LeetCode 题 225 用队列实现栈
 * https://leetcode-cn.com/problems/implement-stack-using-queues/
 *
 * 使用队列实现栈的下列操作：
 * 		push(x) -- 元素 x 入栈
 * 		pop() -- 移除栈顶元素
 * 		top() -- 获取栈顶元素
 * 		empty() -- 返回栈是否为空
 */

type MyStack struct {
	*list.List
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	stack := MyStack{
		list.New(),
	}
	return stack
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int) {
	this.PushBack(x)
	return
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	top := this.Back()
	this.Remove(top)
	return top.Value.(int)
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.Back().Value.(int)
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return this.Len() == 0
}

/*
 * LeetCode T20. 有效的括号
 * https://leetcode-cn.com/problems/valid-parentheses/
 *
 * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
 * 有效字符串需满足：
 * 	左括号必须用相同类型的右括号闭合。
 * 	左括号必须以正确的顺序闭合。
 *  注意空字符串可被认为是有效字符串。
 * 示例 1:
 * 输入: "()"
 * 输出: true
 * 示例 2:
 * 输入: "([)]"
 * 输出: false
 */
// 辅助栈方法
func IsValid(s string) bool {
	if len(s) == 0 {
		return true
	}
	if len(s)%2 == 1 {
		return false
	}
	stack := list.New()
	for _, r := range s {
		if r == '(' {
			stack.PushBack(')')
		} else if r == '[' {
			stack.PushBack(']')
		} else if r == '{' {
			stack.PushBack('}')
		} else {
			if stack.Len() == 0 {
				return false
			}
			top := stack.Back()
			if top.Value.(int32) != r {
				return false
			}
			stack.Remove(top)
		}
	}
	if stack.Len() != 0 {
		return false
	}
	return true
}
