

# 剑指 Offer 

## First Day

### 剑指 Offer 03. 数组中重复的数字

```txt
找出数组中重复的数字。

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        if (nums == null || nums.length <= 0) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                int t = nums[i];
                nums[i] = nums[t];
                nums[t] = t;
            }
        }
        return -1;
    }
}               
```

### 剑指 Offer 04. 二维数组中的查找

```txt
在一个 m * n 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

示例:
现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。
给定 target = 20，返回 false。
```

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = n - 1;
        while (row <= m - 1 && col >= 0) {
            if (matrix[row][col] > target) {
                col--;
            } else if (matrix[row][col] < target) {
                row++;
            } else {
                return true;
            }
        }
        return false;
    }
}
```

### 剑指 Offer 05. 替换空格

```txt
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

示例 1：
输入：s = "We are happy."
输出："We%20are%20happy."
```

```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder();
        for(Character c : s.toCharArray())
        {
            if(c == ' ') res.append("%20");
            else res.append(c);
        }
        return res.toString();
    }
}
```

```java
class Solution {
    public String replaceSpace(String s) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == ' ') {
                count++;
            }
        }
        char[] ch = new char[s.length() + count * 2];
        int p1 = s.length() - 1;
        int p2 = ch.length - 1;
        while (p1 >= 0 && p2 >= 0) {
            char c = s.charAt(p1--);
            if (c == ' ') {
                ch[p2--] = '0';
                ch[p2--] = '2';
                ch[p2--] = '%';
            } else {
                ch[p2--] = c;
            }
        }
        return new String(ch);
    }
}
```



### 剑指 Offer 06. 从尾到头打印链表

```txt

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

示例 1：
输入：head = [1,3,2]
输出：[2,3,1]
```

```java
class Solution {
    public int[] reversePrint(ListNode head) {
        int count = 0;
        ListNode cur = head;
        while (cur != null) {
            cur = cur.next;
            count++;
        }
        int[] res = new int[count];
        cur = head;
        for (int i = 0; i < count; i++) {
            res[count - 1 - i] = cur.val;
            cur = cur.next;
        }
        return res;
    }
}
```

```java
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ListNode head = new ListNode(-1);
        while (listNode != null) {
            ListNode next = listNode.next;
            listNode.next = head.next;
            head.next = listNode;
            listNode = next;
        }
        ArrayList<Integer> res = new ArrayList<>();
        while (head.next != null) {
            res.add(head.next.val);
            head = head.next;
        }
        return res;
    }
}
```

```java
public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
    ArrayList<Integer> ret = new ArrayList<>();
    if (listNode != null) {
        ret.addAll(printListFromTailToHead(listNode.next));
        ret.add(listNode.val);
    }
    return ret;
}
```

### 剑指 Offer 07. 重建二叉树

```txt
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如，给出
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7
```

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || inorder == null || preorder.length == 0 || inorder.length == 0)
            return null;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }
        return helper(preorder, 0, preorder.length - 1, map, 0, inorder.length - 1);
    }
    public TreeNode helper(int[] preorder, int preLeft, int preRight, Map<Integer, Integer> map, int inLeft, int inRight) {
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        int rootVal = preorder[preLeft];
        TreeNode root = new TreeNode(rootVal);
        int index = map.get(rootVal);
        root.left = helper(preorder, preLeft + 1, index - inLeft + preLeft, map, inLeft, index - 1);
        root.right = helper(preorder, index - inLeft + preLeft + 1, preRight, map, index + 1, inRight);
        return root;
    }
}
```

### 剑指 Offer 08.二叉树的下一个结点

```txt
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
public class TreeLinkNode {
 
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;
 
    TreeLinkNode(int val) {
        this.val = val;
    }
}
```

```java
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode)
    {
       if (pNode.right != null) {
           TreeLinkNode node = pNode.right;
           while (node.left != null) 
               node = node.left;
           return node;
       } else {
           while (pNode.next != null) {
               TreeLinkNode parent = pNode.next;
               if (parent.left == pNode) {
                   return parent;
               }
               pNode = pNode.next;
           }
       }
        return null;
    }
}
```

### 剑指 Offer 09. 用两个栈实现队列

```txt
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 ) 

示例 1：
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]

示例 2：
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```

```java
class CQueue {
    LinkedList<Integer> A, B;

    public CQueue() {
        A = new LinkedList();
        B = new LinkedList();
    }
    
    public void appendTail(int value) {
        //相当于Stack的push(),add()操作
        A.addLast(value); 
    }
    
    public int deleteHead() {
        if (!B.isEmpty()) {
            //相当于Stack的pop()操作
            return B.removeLast();
        }
        if (A.isEmpty()) {
            return -1;
        }
        while (!A.isEmpty()) {
            B.addLast(A.removeLast());
        }
        return B.removeLast();
    }
}
```

```java
class CQueue {
    Stack<Integer> in, out;
    public CQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }
    
    public void appendTail(int value) {
        in.push(value);
    }
    
    public int deleteHead() {
        if (out.isEmpty()) {
            while (!in.isEmpty()) {
                out.push(in.pop());
            }
        }
        return out.isEmpty() ? -1 : out.pop();
    }
}
```



## Second Day

### 剑指 Offer 10.1.斐波那契数列

```txt
写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
示例 1：
输入：n = 2
输出：1

示例 2：
输入：n = 5
输出：5
```

```java
class Solution {
    public int fib(int n) {
        int a = 0, b = 1, sum;
        for(int i = 0; i < n; i++){
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
}
public class Solution {
    public int Fibonacci(int n) {
        if (n <= 1)
            return n;
        int a = 0, b = 1;
        int sum = 0;
        for (int i = 2; i <= n; i++) {
            sum = a + b;
            a = b;
            b = sum;
        }
        return sum;
    }
}
```

### 剑指 Offer 10.2 矩形覆盖

```txt
我们可以用 2*1 的小矩形横着或者竖着去覆盖更大的矩形。请问用 n 个 2*1 的小矩形无重叠地覆盖一个 2*n 的大矩形，总共有多少种方法？
```

```java
public class Solution {
    public int RectCover(int n) {
        if (n <= 2)
            return n;
        int res = 0, a= 1, b = 2;
        for (int i = 3; i <= n; i++) {
            res = a + b;
            a = b;
            b = res;
        }
        return res;
    }
}
```

### 剑指 Offer 10.3. 青蛙跳台阶问题

```txt
一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：
输入：n = 2
输出：2

示例 2：
输入：n = 7
输出：21

示例 3：
输入：n = 0
输出：1
```

```java
class Solution {
    public int numWays(int n) {
        int a = 1, b = 2, sum;
        for (int i = 1; i < n; i++) {
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
}
public class Solution {
    public int JumpFloor(int n) {
        if (n <= 2)
            return n;
        int res = 0, a= 1, b = 2;
        for (int i = 2; i < n; i++) {
            res = a + b;
            a = b;
            b = res;
        }
        return res;
    }
}
```

### 剑指 Offer 10.4. 变态跳台阶问题

```txt
一只青蛙一次可以跳上 1 级台阶，也可以跳上 2 级... 它也可以跳上 n 级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
```

```java
public class Solution {
    public int JumpFloorII(int target) {
        if (target <= 0) {
            return -1;
        } else if (target == 1) {
            return 1;
        } else {
            return 2 * JumpFloorII(target - 1);
        }
    }
}
public class Solution {
    public int JumpFloorII(int target) {
        return 1 << (target - 1);
    }
}
```

### 剑指 Offer 11. 旋转数组的最小数字

```txt
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

示例 1：
输入：[3,4,5,1,2]
输出：1

示例 2：
输入：[2,2,2,0,1]
输出：0
```

```java
class Solution {
    public int minArray(int[] numbers) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int mid = i + (j - i) / 2;
            if (numbers[mid] > numbers[j]) {
                i = mid + 1;
            } else if (numbers[mid] < numbers[j]) {
                j = mid;
            } else {
                j--;
            }
        }
        return numbers[i];
    }
}
```

### 剑指 Offer 12. 矩阵中的路径

```txt
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]
但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

示例 1：
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true

示例 2：
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        char[] words = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, words, i, j, 0))
                    return true;
            }
        }
        return false;
    }
    boolean dfs(char[][] board, char[] words, int i, int j, int id) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != words[id])
            return false;
        if (id == words.length - 1)
            return true;
        char tmp = board[i][j];
        board[i][j] = '#';
        boolean flag = dfs(board, words, i + 1, j, id + 1) ||
                       dfs(board, words, i - 1, j, id + 1) ||
                       dfs(board, words, i, j + 1, id + 1) ||
                       dfs(board, words, i, j - 1, id + 1);
        board[i][j] = tmp;
        return flag;
    }
}
```

### 剑指 Offer 13. 机器人的运动范围

```txt
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

示例 1：
输入：m = 2, n = 3, k = 1
输出：3

示例 2：
输入：m = 3, n = 1, k = 0
输出：1
```

```java
class Solution {
    public int movingCount(int m, int n, int k) {
        boolean[][] vis = new boolean[m][n];
        int res = dfs(m, n, k, 0, 0, vis);
        return res;
    }
    int isSum(int n) {
        int sum = 0;
        while (n != 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }
    int dfs(int m, int n, int k, int i, int j, boolean[][] vis) {
        if (i < 0 || i >= m || j < 0 || j >= n || isSum(i) + isSum(j) > k || vis[i][j] == true)
            return 0;
        vis[i][j] = true;
        return 1 + dfs(m, n, k, i + 1, j, vis) + dfs(m, n, k, i, j + 1, vis) ;
    } 
}
```

### 剑指 Offer 14. 剪绳子

```txt
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1

示例 2:
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

```java
class Solution {
    public int cuttingRope(int n) {
        if (n <= 3)
            return n-1;
        int a = n / 3, b = n % 3;
        if (b == 0)
            return (int) Math.pow(3, a);
        if (b == 1)
            return (int) Math.pow(3, a - 1) * 4;
        return (int) Math.pow(3, a) * 2;
    }
}
class Solution {
    public int cuttingRope(int n) {
        if (n <= 3)
            return n-1;
        long res = 1;
        while (n > 4) {
            res *= 3;
            res %= 1000000007;
            n -= 3;
        }
        return (int) res * n % 1000000007;
    }
}
```

### 剑指 Offer 15. 二进制中1的个数

```txt
请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

示例 1：
输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。

示例 2：
输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```

```java
public class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= n - 1;
            count++;
        }
        return count;
    }
}
```

## Third Day

### 剑指 Offer 16. 数值的整数次方

```txt
实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。

示例 1:
输入: 2.00000, 10
输出: 1024.00000

示例 2:
输入: 2.10000, 3
输出: 9.26100
```

```java
public class Solution {
    public double Power(double base, int exponent) {
       if (exponent == 0)
           return 1;
        if (exponent == 1)
            return base;
        boolean isNegative = false;
        if (exponent < 0) {
            isNegative = true;
            exponent = -exponent;
        }
        double pow = Power(base * base, exponent / 2);
        if (exponent % 2 != 0) 
            pow *= base;
        return isNegative ? 1 / pow : pow;
  }
}
class Solution {
    public double myPow(double x, int n) {
        double res = 1.0;
        long b = n;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1)
                res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }
}
```

### 剑指 Offer 17. 打印从1到最大的n位数

```txt
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数即 999。
```

```java
class Solution {
    public int[] printNumbers(int n) {
        int max = 0;
        for (int i = 0; i < n; i++) {
            max = max * 10 + 9;
        }
        //int max = (int)Math.pow(10, n) - 1;
        int[] res = new int[max];
        for (int i = 0; i < max; i++) {
            res[i] = i + 1;
        }
        return res;
    }
}
```

### 剑指 Offer 18.1 在 O(1) 时间内删除链表节点

```txt
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
返回删除后的链表的头节点。
注意：此题对比原题有改动
示例 1:
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

示例 2:
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

```java
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        if (head.val == val) {
            return head.next;
        }
        ListNode pre = head, cur = head.next;
        while (cur != null && cur.val != val) {
            pre = cur;
            cur = cur.next;
        }
        if (cur != null) {
            pre.next = cur.next;
        }
        return head;
    }
}
```

### 剑指 Offer 18.2 删除链表中重复的节点

```txt
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 
例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
```

```java
public class Solution {
    public ListNode deleteDuplication(ListNode pHead) {
		if (pHead == null || pHead.next == null)
            return pHead;
        ListNode next = pHead.next;
        if (next.val == pHead.val) {
            while (next != null && next.val == pHead.val)
                next = next.next;
            return deleteDuplication(next);
        } else {
            pHead.next = deleteDuplication(pHead.next);
            return pHead;
        }
    }
}
```

### 剑指Offer 19. 正则表达式匹配

```java
class Solution {
    public boolean isMatch(String s, String p) {
        if (p.isEmpty())
            return s.isEmpty();
        
        boolean first_match = !s.isEmpty() && (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.');

        if (p.length() >= 2 && p.charAt(1) == '*') {
            return (isMatch(s, p.substring(2)) || (first_match && isMatch(s.substring(1), p)));
        } else {
            return first_match && isMatch(s.substring(1), p.substring(1));
        }

        
    }
}
```

```java
class Solution {
    public boolean isMatch(String A, String B) {
        int n = A.length();
        int m = B.length();
        boolean[][] f = new boolean[n + 1][m + 1];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                //分成空正则和非空正则两种
                if (j == 0) {
                    f[i][j] = i == 0;
                } else {
                    //非空正则分为两种情况 * 和 非*
                    if (B.charAt(j - 1) != '*') {
                        if (i > 0 && (A.charAt(i - 1) == B.charAt(j - 1) || B.charAt(j - 1) == '.')) {
                            f[i][j] = f[i - 1][j - 1];
                        }
                    } else {
                        //碰到 * 了，分为看和不看两种情况
                        //不看
                        if (j >= 2) {
                            f[i][j] = f[i][j - 2];
                        }
                        //看
                        if (i >= 1 && j >= 2 && 
                            (A.charAt(i - 1) == B.charAt(j - 2) || B.charAt(j - 2) == '.')) {
                            f[i][j] |= f[i - 1][j];
                        }
                    }
                }
            }
        }
        return f[n][m];
    }
}

```

### 剑指Offer 20. 表示数值的字符串

```java
class Solution {
    private int index = 0;//全局索引
    private boolean scanUnsignedInteger(String str) {
        //是否包含无符号数
        int before = index;
        while(str.charAt(index) >= '0' && str.charAt(index) <= '9') 
            index++;
        return index > before;
    }
    private boolean scanInteger(String str) {
        //是否包含有符号数
        if(str.charAt(index) == '+' || str.charAt(index) == '-') 
               index++;
        return scanUnsignedInteger(str);
    }
    public boolean isNumber(String s) {
        //空字符串
        if(s == null || s.length() == 0)
            return false;
        //添加结束标志
        s = s + '|';
        //跳过首部的空格
        while(s.charAt(index) == ' ')
            index++;
        boolean numeric = scanInteger(s); //是否包含整数部分
        if(s.charAt(index) == '.') {  
            index++;
            //有小数点，处理小数部分
            //小数点两边只要有一边有数字就可以，所以用||，
            //注意scanUnsignedInteger要在前面，否则不会进
            numeric = scanUnsignedInteger(s) || numeric;
        }
        if((s.charAt(index) == 'E' || s.charAt(index) == 'e')) { 
            index++;
            //指数部分
            //e或E的两边都要有数字，所以用&&
            numeric = numeric && scanInteger(s);
        }
        //跳过尾部空格
        while(s.charAt(index) == ' ')
            index++;
        return numeric && s.charAt(index) == '|' ;
    }
}
```

### 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面

```txt
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

示例：
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

```java
class Solution {
    public int[] exchange(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            while (l < r && nums[l] % 2 == 1) {
                l++;
            }
            while (l < r && nums[r] % 2 == 0) {
                r--;
            }
            int t = nums[r];
            nums[r] = nums[l];
            nums[l] = t;
        }
        return nums;
    }
}
class Solution {
    public int[] exchange(int[] nums) {
        int low = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[fast] % 2 == 1) {
                int t = nums[fast];
                nums[fast] = nums[low];
                nums[low] = t;
                low++;
            }
            fast++;
        }
        return nums;
    }
}
```

### 剑指 Offer 22. 链表中倒数第k个节点

```txt
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

示例：
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null) return null;
        ListNode cur = head;
        while (cur != null && k > 0) {
            cur = cur.next;
            k--;
        }
        if (k > 0) return null;
        while (cur != null) {
            head = head.next;
            cur = cur.next;
        }
        return head;
    }
}
```

### 剑指 Offer 22. 链表中环的入口结点

```txt
一个链表中包含环，请找出该链表的环的入口结点。要求不能使用额外的空间。
```

```java
public class Solution {
    public ListNode EntryNodeOfLoop(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode slow = head.next, fast = head.next.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            fast = fast.next.next;
            slow = slow.next;
        }
        fast = head;
        while (slow != fast) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }   
}
```

### 剑指 Offer 24. 反转链表

```txt
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode dummy = new ListNode(-1);
        while (head != null) {
            ListNode next = head.next;
            head.next = dummy.next;
            dummy.next = head;
            head = next;
        }
        return dummy.next;
    }
}
public ListNode ReverseList(ListNode head) {
    if (head == null || head.next == null)
        return head;
    ListNode next = head.next;
    head.next = null;
    ListNode newHead = ReverseList(next);
    next.next = head;
    return newHead;
}
```

## Fourth Day

### 剑指 Offer 25. 合并两个排序的链表

```java
//递归
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
//迭代
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 != null) {
            cur.next = l1;
        }
        if (l2 != null) {
            cur.next = l2;
        }
        return dummy.next;
    }
}
```

### 剑指 Offer 26. 树的子结构

```txt
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

     3
    / \
   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：
输入：A = [1,2,3], B = [3,1]
输出：false

示例 2：
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```

```java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) 
            return false;
       	return isSubTreeWithRoot(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
    public boolean isSubTreeWithRoot(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null) return false;
        if (A.val != B.val) return false;
        return isSubTreeWithRoot(A.left, B.left) && isSubTreeWithRoot(A.right, B.right);
    }
}
```

### 剑指 Offer 27. 二叉树的镜像

```txt
请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

     4
   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1

示例 1：
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
		if (root == null) return null;
        swap(root);
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }
    public void swap(TreeNode root) {
        TreeNode t = root.left;
        root.left = root.right;
        root.right = t;
    }
}
```

### 剑指 Offer 28. 对称的二叉树

```txt
请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3

示例 1：
输入：root = [1,2,2,3,4,4,3]
输出：true

示例 2：
输入：root = [1,2,2,null,3,null,3]
输出：false
```

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
		if (root == null) return true;
        return isSymmetric(root.left, root.right);
    }
    public boolean isSymmetric(TreeNode t1, TreeNode t2) {
		if (t1 == null && t2 == null)
            return true;
        if (t1 == null || t2 == null)
            return false;
        if (t1.val != t2.val)
            return false;
        return isSymmetric(t1.left, t2.right) && isSymmetric(t1.right, t2.left);
    }
}
```

### 剑指 Offer 29. 顺时针打印矩阵

```java
class Solution {
    int index = 0;
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return new int[0];
        int m = matrix.length, n = matrix[0].length;
        int[] res = new int[m * n];
        int aR = 0, aC = 0;
        int bR = m - 1, bC = n - 1;
        while (aR <= bR && aC <= bC) {
            saveNum(res, matrix, aR++, aC++, bR--, bC--);
        }
        return res;
    }
    private void saveNum(int[] res, int[][] matrix, int aR, int aC, int bR, int bC) {
        if (aR == bR) {
            for (int i = aC; i <= bC; i++) {
                res[index++] = matrix[aR][i];
            }
        } else if (aC == bC) {
            for (int i = aR; i <= bR; i++) {
                res[index++] = matrix[i][aC];
            }
        } else {
            int curR = aR, curC = aC;
            while (curC != bC) {
                res[index++] = matrix[aR][curC++];
            }
            while (curR != bR) {
                res[index++] = matrix[curR++][bC];
            }
            while (curC != aC) {
                res[index++] = matrix[bR][curC--];
            }
            while (curR != aR) {
                res[index++] = matrix[curR--][aC];
            }
        } 
    }
}
```

### 剑指 Offer 30. 包含min函数的栈

```java
class MinStack {
	Stack<Integer> dataStack, minStack;
    /** initialize your data structure here. */
    public MinStack() {
		dataStack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int x) {
		dataStack.push(x);
        minStack.push(minStack.isEmpty() ? x : Math.min(x, minStack.peek()));
    }
    
    public void pop() {
		dataStack.pop();
        minStack.pop();
    }
    
    public int top() {
		return dataStack.peek();
    }
    
    public int min() {
		return minStack.peek();
    }
}
```

```java
class MinStack {
    Stack<Integer> A, B;
    public MinStack() {
        A = new Stack<>();
        B = new Stack<>();
    }
    public void push(int x) {
        A.add(x);
        if(B.empty() || B.peek() >= x)
            B.add(x);
    }
    public void pop() {
        if(A.pop().equals(B.peek()))
            B.pop();
    }
    public int top() {
        return A.peek();
    }
    public int min() {
        return B.peek();
    }
}
```

### 剑指Offer 31.栈的压入、弹出序列

```txt
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
		Stack<Integer> stack = new Stack<>();
        int i = 0;
        for (int num : pushed) {
            stack.push(num);
            while (!stack.isEmpty() && stack.peek() == popped[i]) {
                stack.pop();
                i++;
            }
        }
        return stack.isEmpty();
    }
}
```

### 剑指 Offer 32 - I. 从上到下打印二叉树

```txt
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回：
[3,9,20,15,7]
```

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
		List<Integer> ans = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root == null) 
            return new int[0];
        queue.add(root);
        while (!queue.isEmpty()) {
            root = queue.poll();
            ans.add(root.val);
            if (root.left != null)
                queue.add(root.left);
           	if (root.right != null)
                queue.add(root.right);
        }
        int[] res = new int[ans.size()];
        for (int i = 0; i < res.length; i++)
            res[i] = ans.get(i);
        return res;
    }
}
```

### 剑指 Offer 32 - II. 从上到下打印二叉树

```txt

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：
[
  [3],
  [9,20],
  [15,7]
]
```

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root == null) return res;
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                root = queue.poll();
                level.add(root.val);
                if (root.left != null)
                    queue.add(root.left);
               	if (root.right != null)
                    queue.add(root.right);
            }
            res.add(level);
        }
        return res;
    }
}
```

### 剑指 Offer 32 - III. 从上到下打印二叉树

```txt
请实现一个函数按照“之”字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [20,9],
  [15,7]
]
```

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if (root == null) return res;
        queue.add(root);
        boolean reverse = false;
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                root = queue.poll();
                level.add(root.val);
                if (root.left != null)
                    queue.add(root.left);
               	if (root.right != null)
                    queue.add(root.right);
            }
            if (reverse)
                Collections.reverse(level);
            reverse = !reverse;
            res.add(level);
        }
        return res;
    }
}
```

## Fifth Day

### 剑指 Offer 33. 二叉搜索树的后序遍历序列

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return helper(postorder, 0, postorder.length - 1);
    }
    boolean helper(int[] postorder, int i, int j) {
        if (i >= j) return true;
        int p = i;
        while (postorder[p] < postorder[j])
            p++;
        int m = p;
        while (postorder[p] > postorder[j])
            p++;
        return p == j && helper(postorder, i, m - 1) && helper(postorder, m, j - 1);
    }
}
```

```java
public class Solution {
    public boolean VerifySquenceOfBST(int [] postorder) {
        if (postorder == null || postorder.length == 0)
            return false;
        return helper(postorder, 0, postorder.length - 1);
    }
    boolean helper(int[] postorder, int first, int last) {
        if (first >= last) return true;
        int rootVal = postorder[last];
        int p = first;
        while (postorder[p] < rootVal)
            p++;
        for (int i = p; i < last; i++) {
            if (postorder[i] < rootVal)
                return false;
        }
        return helper(postorder, first, p - 1) && helper(postorder, p, last - 1);
    }
}
```

### 剑指Offer 34.二叉树中和为某一值的路径

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        dfs(root, sum);
        return res;
    }
    void dfs(TreeNode root, int sum) {
        if (root == null) return;
        path.add(root.val);
        sum -= root.val;
        if (sum == 0 && root.left == null && root.right == null)
            res.add(new ArrayList<>(path));
        dfs(root.left, sum);
        dfs(root.right, sum);
        path.remove(path.size() - 1);
    }
}
```

### 剑指Offer 35.复杂链表的复制

```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;

        // 1.clone链表
        Node cur = head;
        while (cur != null) {
            Node clone = new Node(cur.val);
            clone.next = cur.next;
            cur.next = clone;
            cur = clone.next;
        }

        // 2.建立random的链接
        cur = head;
        while (cur != null) {
            Node clone = cur.next;
            if (cur.random != null)
                clone.random = cur.random.next;
            cur = clone.next;
        }

        // 3.拆分clone链表
        cur = head;
        Node cloneHead = cur.next;
        while (cur.next != null) {
            Node next = cur.next;
            cur.next = next.next;
            cur = next;
        }

        return cloneHead;
    }
}
```

### 剑指Offer36.二叉搜索树与双向链表

```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    Node pre, head;

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;

        dfs(root);
        pre.right = head;
        head.left = pre;

        return head;
    }

    void dfs(Node cur) {
        if (cur == null) return;
        dfs(cur.left);

        // 双向链表的指向问题
        if (pre == null)
            head = cur;
        else
            pre.right = cur;
        
        cur.left = pre;
        pre = cur;

        dfs(cur.right);
    }
} 
```

### 剑指Offer37.序列化二叉树

```java
public class Codec {
    public String serialize(TreeNode root) {
        if(root == null) return "[]";
        StringBuilder res = new StringBuilder("[");
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(node != null) {
                res.append(node.val + ",");
                queue.add(node.left);
                queue.add(node.right);
            }
            else res.append("null,");
        }
        res.deleteCharAt(res.length() - 1);
        res.append("]");
        return res.toString();
    }

    public TreeNode deserialize(String data) {
        if(data.equals("[]")) return null;
        String[] vals = data.substring(1, data.length() - 1).split(",");
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(!vals[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.left);
            }
            i++;
            if(!vals[i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(vals[i]));
                queue.add(node.right);
            }
            i++;
        }
        return root;
    }
}
```

### 剑指 Offer 38. 字符串的排列

```java
class Solution {
    public String[] permutation(String s) {
        char[] arr = s.toCharArray();
        List<String> res = new ArrayList<>();

        dfs(res, arr, 0);

        return res.toArray(new String[res.size()]);
    }

    void dfs(List<String> res, char[] arr, int cur) {
        if (cur == arr.length) {
            res.add(String.valueOf(arr));
            return;
        }
        for (int i = cur; i < arr.length; i++) {
            if (canSwap(arr, cur, i)) {
                swap(arr, cur, i);
                dfs(res, arr, cur + 1);
                swap(arr, cur, i);
            }
        }
    }

    boolean canSwap(char[] arr, int start, int end) {
        for (int i = start; i < end; i++) {
            if (arr[i] == arr[end])
                return false;
        }
        return true;
    }

    void swap(char[] arr, int i, int j) {
        char t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
    }
}
```

### 剑指Offer39. 数组中出现次数超过一半的数字

```java
class Solution {
    public int majorityElement(int[] nums) {
        if (nums.length == 0) return nums[0];
        int major = nums[0], count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (count == 0) {
                major = nums[i];
                count = 1;
            } else {
                count += (major == nums[i]) ? 1 : -1;
            }
        }
        return major;
    }
}
```

### 剑指 Offer 40. 最小的k个数

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if(arr.length == 0 || k == 0)
            return new int[0];
        
        return quicksort(arr, 0, arr.length-1, k-1);
    }

    int[] quicksort(int[] num, int start, int end, int k) {
        int j = parition(num, start, end);
        if(j == k) 
            return Arrays.copyOf(num, j + 1);
        else if(j < k)
            return quicksort(num, j+1, end, k);
        else
            return quicksort(num, start, j-1, k);
    }

    int parition(int num[], int start, int end) {
        int tmp = num[start];
        int i = start, j = end;
        while(i < j) {
            while(num[j] >= tmp && j > i) j--;
            if(j > i)
                num[i] = num[j];

            while(num[i] <= tmp && j > i) i++;
            if(j > i)
                num[j] = num[i];
        }

        num[i] = tmp;
        return i;
    }
}
```

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }

        int[] res = new int[k];
        PriorityQueue<Integer> pq = new PriorityQueue<>((o1, o2) -> (o2 - o1));
        
        for (int i = 0; i < arr.length; i++) {
            pq.offer(arr[i]);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        for (int i = 0; i < k; i++) {
            res[i] = pq.poll();
        }
        return res;
    }
}
```

## Sixth Day

### 剑指Offer41. 数据流中的中位数

```java
class MedianFinder {
    PriorityQueue<Integer> maxHeap;
    PriorityQueue<Integer> minHeap;
    int count = 0;
    
    public MedianFinder() {
        maxHeap = new PriorityQueue<>((x, y) -> y - x);
        minHeap = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        maxHeap.add(num);
        count++;
        minHeap.add(maxHeap.poll());
        if (count % 2 == 1) {
            maxHeap.add(minHeap.poll());
        }
    }
    
    public double findMedian() {
        if (count % 2 == 1) {
            return maxHeap.peek();
        } else {
            return (minHeap.peek() + maxHeap.peek()) / 2.0;
        }
    }
}
```

### 剑指Offer42. 连续子数组的最大和

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(nums[i-1], 0);
            res = Math.max(res, nums[i]);
        }
        return res;
    }
}
```

### 剑指Offer43. 1~n整数中1出现的次数

```java
class Solution {
    public int countDigitOne(int n) {
        int digit = 1, res = 0;
        int high = n / 10, cur = n % 10, low = 0;
        while(high != 0 || cur != 0) {
            if(cur == 0) res += high * digit;
            else if(cur == 1) res += high * digit + low + 1;
            else res += (high + 1) * digit;
            low += cur * digit;
            cur = high % 10;
            high /= 10;
            digit *= 10;
        }
        return res;
    }
}
```

### 剑指Offer44. 数字序列中某一位的数字

```java
class Solution {
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit; // 2.
        return Long.toString(num).charAt((n - 1) % digit) - '0'; // 3.
    }
}
```

### 剑指Offer45. 把数组排成最小的数

```java
class Solution {
    public String minNumber(int[] nums) {
        if (nums == null || nums.length == 0) return "";
        String[] ans = new String[nums.length];
        for (int i = 0; i < nums.length; i++)
            ans[i] = String.valueOf(nums[i]);
        Arrays.sort(ans, (s1, s2) -> (s1 + s2).compareTo(s2 + s1));
        StringBuilder res = new StringBuilder();
        for (String str : ans)
            res.append(str);
        return res.toString();
    }
}
```

### 剑指 Offer 46. 把数字翻译成字符串

```java
class Solution {
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int len = s.length();
        if (len < 2) return len;
        char[] arr = s.toCharArray();

        int[] dp = new int[len+1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 1; i < len; i++) {
            dp[i+1] = dp[i];
            int curNum = 10 * (arr[i-1] - '0') + (arr[i] - '0');
            if (curNum > 9 && curNum < 26) {
                dp[i+1] = dp[i] + dp[i-1];
            }
        }
        return dp[len];
    }
}
```

### 剑指 Offer 47. 礼物的最大价值

```java
class Solution {
    public int maxValue(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        
        for (int i = 1; i < m; i++) {
            grid[i][0] += grid[i-1][0];
        }
        for (int j = 1; j < n; j++) {
            grid[0][j] += grid[0][j-1];
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] += Math.max(grid[i-1][j], grid[i][j-1]);
            }
        }
        return grid[m-1][n-1];
    }
}
```

### 剑指Offer48. 最长不含重复字符的子字符串

```txt
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int start = 0, end = 0;
        int res = 0;
        while (end < s.length()) {
            char c = s.charAt(end++);
            while (set.contains(c)) {
                set.remove(s.charAt(start++));
            }
            set.add(c);
            res = Math.max(res, end - start);
        }
        return res;
    }
}
```

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int start = 0, end = 0;
        int res = 0;
        while (end < s.length()) {
            if (map.containsKey(s.charAt(end))) 
                start = Math.max(start, map.get(s.charAt(end)) + 1);
            map.put(s.charAt(end), end++);
            res = Math.max(res, end - start);
        }
        return res;
    }
}
```

### 剑指Offer49. 丑数

```java
class Solution {
    public int nthUglyNumber(int n) {
        int a = 0, b = 0, c = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            int n2 = dp[a]*2, n3 = dp[b]*3, n5 = dp[c]*5;
            dp[i] = Math.min(Math.min(n2, n3), n5);
            if (dp[i] == n2) a++;
            if (dp[i] == n3) b++;
            if (dp[i] == n5) c++;
        }
        return dp[n-1];
    }
}
```



### 剑指Offer50. 第一个只出现一次的字符

```java
class Solution {
    public char firstUniqChar(String s) {
        int[] ints = new int[256];
        char[] chars = s.toCharArray();
        for (char c : chars) {
            ints[c]++;
        }
        for (char c : chars) {
            if (ints[c] == 1)
                return c;
        }
        return ' ';
    }
}
```

```java
class Solution {
    public char firstUniqChar(String s) {
        Map<Character, Boolean> map = new HashMap<>();
        char[] chars = s.toCharArray();
        for (char c : chars) {
            map.put(c, !map.containsKey(c));
        }
        for (char c : chars) {
            if(map.get(c))
                return c;
        }
        return ' ';
    }
}
```

## Seventh Day

### 剑指Offer52. 数组中的逆序对

```txt
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

示例 1:
输入: [7,5,6,4]
输出: 5
```

```java
public class Solution {

    public int reversePairs(int[] nums) {
        int cnt = 0;
        int len = nums.length;
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                if (nums[i] > nums[j]) {
                    cnt++;
                }
            }
        }
        return cnt;
    }
}
```

```java
class Solution {
    public int reversePairs(int[] nums) {
        int len = nums.length;

        if (len < 2) {
            return 0;
        }
        int[] tmp = new int[len];
        return reversePairs(nums, 0, len - 1, tmp);
    }

    public int reversePairs(int[] nums, int left, int right, int[] tmp) {
        if (left == right) {
            return 0;
        }
        int mid = left + (right - left) / 2;
        int leftPairs = reversePairs(nums, left, mid, tmp);
        int rightPairs = reversePairs(nums, mid + 1, right, tmp);

        if (nums[mid] <= nums[mid + 1]) {
            return leftPairs + rightPairs;
        }

        int crossPairs = mergeCount(nums, left, mid, right, tmp);
        return leftPairs + rightPairs + crossPairs;
    }

    public int mergeCount(int[] nums, int left, int mid, int right, int[] tmp) {
        for (int i = left; i <= right; i++) {
            tmp[i] = nums[i];
        }
        int i = left, j = mid + 1, count = 0;

        for (int k = left; k <= right; k++) {
            if (i == mid + 1) {
                nums[k] = tmp[j++];
            } else if (j == right + 1) {
                nums[k] = tmp[i++];
            } else if (tmp[i] <= tmp[j]) {
                nums[k] = tmp[i++];
            } else {
                nums[k] = tmp[j++];
                count += mid - i + 1;
            }
        }
        
        return count;
    }
}
```

### 剑指Offer52. 两个链表的第一个公共节点

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA, p2 = headB;
        while (p1 != p2) {
            p1 = (p1 == null) ? headB : p1.next;
            p2 = (p2 == null) ? headA : p2.next;
        }
        return p1;
    }
}
```

### 剑指Offer53 I. 在排序数组中查找数字 

```txt
统计一个数字在排序数组中出现的次数。
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

```java
//二分法找到第一次数字出现的位置findFirstPosition(), 返回findFirstPosition(target+1) - findFirstPosition(target);
class Solution {
    public int search(int[] nums, int target) {
        return findFirstPosition(nums, target+1) - findFirstPosition(nums, target);
    }
    int findFirstPosition(int[] nums, int target) {
        int low = -1, high = nums.length;
        while (low + 1 != high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] >= target)
                high = mid;
            else
                low = mid;
        }
        //if (high != nums.length && nums[high] == target) {
        //	return high;
    	//}
        return high;
    }
}
```

```java
class Solution {
    public int search(int[] nums, int target) {
        return findFirstPosition(nums, target+1) - findFirstPosition(nums, target);
    }
    int findFirstPosition(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] < target)
                low = mid + 1;
            else
                high = mid - 1;
        }
        //if (low != nums.length && nums[low] == target) {
        //	return low;
    	//}
        return low;
    }
}
```



### 剑指 Offer 53 - II. 0～n-1中缺失的数字

```txt
一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

示例 1:
输入: [0,1,3]
输出: 2
```

```java
class Solution {
    public int missingNumber(int[] nums) {
        int low = 0, high = nums.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == mid)
                low = mid + 1;
            else
                high = mid - 1;
        }
        return low;
    }
}
```

### 剑指Offer54. 二叉搜索树的第K大节点

```java
class Solution {
	int res = 0, cnt = 0;

    public int kthLargest(TreeNode root, int k) {
		inOrder(root, k);
		return res;
    }

	void inOrder(TreeNode root, int k) {
		if (root == null) return;
		inOrder(root.right, k);
		if (++cnt == k) {
			res = root.val;
			return;
		}
		inOrder(root.left, k);
	}
}
```

### 剑指Offer55 I. 二叉树的最大深度

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int left = maxDepth(root.left) + 1;
        int right = maxDepth(root.right) + 1;
        return Math.max(left, right);
    }
}
```

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        //List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int depth = 0;
        
        while (!queue.isEmpty()) {
            int size = queue.size();
            //List<Integer> level = new ArrayList<>();
            for (int i = 0; i < size; i++) {
            	TreeNode node = queue.poll();
                //level.add(node.val);
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);  
            }
            //res.add(level);
            depth++;
        }
        
        return depth;
    }
}
```

### 剑指Offer55 II. 平衡二叉树

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        int left = depth(root.left);
        int right = depth(root.right);
        if (Math.abs(left - right) > 1)
            return false;
        return isBalanced(root.left) && isBalanced(root.right);
    }

    public int depth(TreeNode root) {
        if (root == null) return 0;
        return Math.max(depth(root.left),depth(root.right)) + 1;
    }
}
```

### 剑指Offer56 I. 数组中只出现一次的数字

```txt
一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

示例 1：
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
    
示例 2：
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
```

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int xor = 0;
        /* 
            0 ^ num = num
            xor = a ^ b
            nums = {2,2,3,4,6,3}
            xor = (2 ^ 2) ^ (3 ^ 3) ^ (4 ^ 6) = 4 ^ 6 = (1000) ^ (1010) = 0010
        */
        for (int num : nums) {
            xor ^= num;
        }

        int mark = 1;
        while ((mark & xor) == 0) {
            mark <<= 1;
        }
        // mark = 0010

        int a = 0, b = 0;
        for (int num : nums) {
            if ((num & mark) == 0) {
                a ^= num;
            } else {
                b ^= num;
            }
        }

        return new int[]{a, b};
    }
}
```

### 剑指 Offer 56 - II. 数组中数字出现的次数 II

```txt
在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

示例 1：
输入：nums = [3,4,3,3]
输出：4

示例 2：
输入：nums = [9,1,7,9,7,9,7]
输出：1
```

```java
class Solution {
    public int singleNumber(int[] nums) {
        int[] tmp = new int[32];
        for (int num : nums) {
            for (int i = 31; i >= 0; i--) {
                tmp[i] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res <<= 1;
            res |= tmp[i] % 3;
        }
        return res;
    }
}
```

### 剑指Offer 57. 和为S的两个数字

```txt
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
示例 2：
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
```

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            
            if (map.containsKey(target - nums[i])) {
                return new int[]{nums[map.get(target - nums[i])], nums[i]};
            }
            map.put(nums[i], i);
        }
        return new int[0];
    }
}
```

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        while (low < high) {
            int sum = nums[low] + nums[high];
            if (sum == target) {
                return new int[]{nums[low], nums[high]};
            } else if (sum < target) {
                low++;
            } else {
                high--;
            }
        }
        return new int[0];
    }
}
```

### 剑指Offer 57 II. 和为S的连续正数序列

```txt
输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

示例 1：
输入：target = 9
输出：[[2,3,4],[4,5]]

示例 2：
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

```java
import java.util.*;
public class Solution {
    public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
       	ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if (sum < 3) return res;
        
        int small = 1, big = 2;
        int middle = (sum + 1) / 2;
        int curNum = small + big;
        
        while (small < middle) {
            if (curNum == sum) {
                save(res, small, big);
            }
            while (curNum > sum && small < middle) {
                curNum -= small;
                small++;
                
                if (curNum == sum) 
                    save(res, small, big);
            }
            big++;
            curNum += big;
        }
        
        return res;
    }
    void save(ArrayList<ArrayList<Integer>> res, int small, int big) {
        ArrayList<Integer> list = new ArrayList<>();
        for (int i = small; i <= big; i++) {
            list.add(i);
        }
        res.add(list);
    }
}
```

```java
class Solution {
    public int[][] findContinuousSequence(int sum) {
        List<int[]> ans = new ArrayList<>();
        int small = 1, big = 2, curNum = 3;
        int middle = (sum + 1) / 2;
        while (small < middle) {
            if (curNum == sum) {
                save(ans, small, big);
            }
            while (curNum > sum && small < middle) {
                curNum -= small;
                small++;

                if (curNum == sum) 
                    save(ans, small, big);
            }
            big++;
            curNum += big;
        }
        return ans.toArray(new int[ans.size()][]);
    }

    void save(List<int[]> ans, int small, int big) {
        int[] tmp = new int[big - small + 1];
        int index = 0;
        for (int i = small; i <= big; i++) {
            tmp[index++] = i;
        }
        ans.add(tmp);
    }
}
```

### 剑指 Offer 58 - I. 翻转单词顺序

```txt
示例 1：
输入: "   the sky is blue"
输出: "blue is sky the"
```

```java
class Solution {
    public String reverseWords(String s) {
		String[] words = s.trim().split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            if (words[i].equals("")) continue;
            sb.append(words[i] + " ");
        }
        return sb.toString().trim();
    }
}
```

### 剑指 Offer 58 - II. 左旋转字符串

```java
class Solution {
    public String reverseLeftWords(String s, int k) {
        char[] chars = s.toCharArray();
        reverse(chars, 0, k - 1);
        reverse(chars, k, chars.length - 1);
        reverse(chars, 0, chars.length - 1);
        return String.valueOf(chars);
    }

    void reverse(char[] chars, int i, int j) {
        while (i < j) {
            char c = chars[i];
            chars[i] = chars[j];
            chars[j] = c;
            i++;
            j--;
        }
    }
}
```

### 剑指Offer 59 I. 滑动窗口的最大值

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k == 0) return new int[0];
        int left = 0, right = k-1, maxIndex = -1, index = 0;
        int[] res = new int[nums.length - k + 1];
        while (right < nums.length) {
            if (maxIndex < left) {
                maxIndex = left;
                for (int i = left + 1; i <= right; i++)
                    maxIndex = nums[i] > nums[maxIndex] ? i : maxIndex;
            } else {
                if (nums[right] >= nums[maxIndex])
                    maxIndex = right;
            }
            left++;
            right++;
            res[index++] = nums[maxIndex];
        }
        return res;
    }
}
```

###    剑指Offer 59 II. 队列的最大值

```java
class MaxQueue {
    Queue<Integer> q;
    Deque<Integer> d;

    public MaxQueue() {
        q = new LinkedList<>();
        d = new LinkedList<>();
    }
    
    public int max_value() {
        if (q.isEmpty()) {
            return -1;
        }
        return d.peekFirst();
    }
    
    public void push_back(int value) {
        while (!d.isEmpty() && d.peekLast() < value)
            d.pollLast();
        q.offer(value);
        d.offerLast(value);
    }
    
    public int pop_front() {
        if (q.isEmpty())
            return -1;
        int ans = q.poll();
        if (ans == d.peekFirst())
            d.pollFirst();
        return ans;
    }
}
```

## Eighth Day

### 剑指 Offer 60. n个骰子的点数

```txt
把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

示例 1:
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```

```java
class Solution {
    public double[] twoSum(int n) {
        int[][] dp = new int[n + 1][6 * n + 1];
        for (int i = 1; i <= 6; i++)
            dp[1][i] = 1;
        for (int i = 2; i <= n; i++) 
            /* 使用 i 个骰子最小点数为 i */
            for (int j = i; j <= 6 * i; j++) 
               for (int k = 1; k <= 6 && k <= j; k++) 
                   dp[i][j] += dp[i - 1][j - k];

        double total = Math.pow(6, n);
        double[] ans = new double[5 * n + 1];
        for(int i = n; i <= 6 * n; i++){
            ans[i - n] = ((double)dp[n][i]) / total;
        }
        return ans;       
    }
}
```

### 剑指 Offer 61. 扑克牌中的顺子

```txt
从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
 
示例 1:
输入: [1,2,3,4,5]
输出: True
 

示例 2:
输入: [0,0,1,2,5]
输出: True
 
限制：
数组长度为 5 
数组的数取值为 [0, 13] .
```

```java
class Solution {
    public boolean isStraight(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int max = 0, min = 14;
        for (int num : nums) {
            if (num == 0) continue;
            max = Math.max(max, num);
            min = Math.min(min, num);
            if (set.contains(num)) return false;
            set.add(num);
        }
        return max - min < 5;
    }
}
```

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int joker = 0;
        Arrays.sort(nums);
        for (int i = 0; i < 4; i++) {
            if (nums[i] == 0) joker++;
            else if (nums[i] == nums[i + 1]) return false;
        }
        return nums[4] - nums[joker] < 5;
    }
}
```

### 剑指 Offer 62. 圆圈中最后剩下的数字

```txt
0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。
例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

示例 1：
输入: n = 5, m = 3
输出: 3
示例 2：
输入: n = 10, m = 17
输出: 2
```

```java
class Solution {
    public int lastRemaining(int n, int m) {
        int p = 0;
        for (int i = 2; i <= n; i++) {
            p = (p + m) % i;
        }
        return p;
    }
}
```

```java
// 寻找递推公式  递归求解   类似数学归纳法
class Solution {
    public int lastRemaining(int n, int m) {
        if (n == 1) {
            return 0;
        }
        int x = lastRemaining(n - 1, m);
        return (m + x) % n;
    }

    
}
```

```java
class Solution {
    public int lastRemaining(int n, int m) {
        ArrayList<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        int idx = 0;
        while (n > 1) {
            idx = (idx + m - 1) % n;
            list.remove(idx);
            n--;
        }
        return list.get(0);
    }
}
```

### 剑指 Offer 63. 股票的最大利润

```txt

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

示例 1:
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

```java
class Solution {
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for(int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
}
```

### 剑指 Offer 64. 求1+2+…+n

```txt
求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

示例 1：
输入: n = 3
输出: 6
示例 2：
输入: n = 9
输出: 45
```

```java
class Solution {
    public int sumNums(int n) {
        if (n == 1) return 1;
        return n + sumNums(n-1);
    }
}
```

```java
class Solution {
    int res = 0;
    public int sumNums(int n) {
    	boolean x = n > 1 && sumNums(n - 1) > 0;
        res += n;
        return res;
    }
}
```

### 剑指 Offer 65. 不用加减乘除做加法

```txt
写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

示例:
输入: a = 1, b = 1
输出: 2
```

```java
class Solution {
    public int add(int a, int b) {
        while (b != 0) {
            int c = (a & b) << 1;
            a = a ^ b;
            b = c;
        }
        return a;
    }
}
```

### 剑指 Offer 66. 构建乘积数组

```txt
给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

示例:
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]
```

```java
class Solution {
    public int[] constructArr(int[] A) {
        int n = A.length;
        int[] B = new int[n];
        for (int i = 0, product = 1; i < n; product *= A[i], i++)       /* 从左往右累乘 */
            B[i] = product;
        for (int i = n - 1, product = 1; i >= 0; product *= A[i], i--)  /* 从右往左累乘 */
            B[i] *= product;
        return B;
    }
}
```

### 剑指 Offer 67. 把字符串转换成整数

```java
class Solution {
    public int strToInt(String str) {
        int res = 0, bndry = Integer.MAX_VALUE / 10;
        int i = 0, sign = 1, len = str.length();
        if (len == 0) return 0;
        while (str.charAt(i) == ' ') {
            if (++i == len) return 0;
        }
        if (str.charAt(i) == '-') sign = -1;
        if (str.charAt(i) == '-' || str.charAt(i) == '+') 
            i++;
        for (int j = i; j < len; j++) {
            if (str.charAt(j) < '0' || str.charAt(j) > '9')
                break;
            if (res > bndry || res == bndry && str.charAt(j) > '7')
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            res = res * 10 + (str.charAt(j) - '0');
        }
        return res * sign;
    }
}
```

### 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        return root;
    }
}
```

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        while (root != null) {
            if (root.val > p.val && root.val > q.val) 
                root = root.left;
            else if (root.val < p.val && root.val < q.val) 
                root = root.right;
            else 
                break;
        }
        return root;
    }
}
```

### 剑指 Offer 68 - II. 二叉树的最近公共祖先

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }
}
```





