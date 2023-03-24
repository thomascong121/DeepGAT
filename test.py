class Solution:
    def maxProfit(self, prices) -> int:
        if len(prices) == 0:
            return 0
        if len(prices) == 1:
            return prices[0]
        profit_0 = 0
        profit_1 = -prices[0]
        for i in range(len(prices)):
            profit_0 = max(profit_0, profit_1 + prices[i])
            profit_1 = max(profit_1, profit_0 - prices[i])
        return profit_0


if __name__ == '__main__':
    sol = Solution()
    test = [7,1,5,3,6,4]#[7,6,4,3,1]#
    print(sol.maxProfit(test))