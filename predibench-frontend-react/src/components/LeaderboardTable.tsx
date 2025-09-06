import { ArrowDown, ChevronDown } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { LeaderboardEntry } from '../api'
import { CompanyDisplay } from './ui/company-display'
import { InfoTooltip } from './ui/info-tooltip'
import { ProfitDisplay } from './ui/profit-display'
import { RedirectButton } from './ui/redirect-button'

type SortKey = 'cumulative_profit' | 'brier_score'

interface LeaderboardTableProps {
  leaderboard: LeaderboardEntry[]
  loading?: boolean
  initialVisibleModels?: number
}

export function LeaderboardTable({
  leaderboard,
  loading = false,
  initialVisibleModels = 10
}: LeaderboardTableProps) {
  const [visibleModels, setVisibleModels] = useState(initialVisibleModels)
  const [sortKey, setSortKey] = useState<SortKey>('cumulative_profit')

  const sortedLeaderboard = useMemo(() => {
    return [...leaderboard].sort((a, b) => {
      switch (sortKey) {
        case 'cumulative_profit':
          // Calculate display scores (rounded to 1 decimal place)
          const aDisplayScore = parseFloat((a.final_cumulative_pnl * 100).toFixed(1))
          const bDisplayScore = parseFloat((b.final_cumulative_pnl * 100).toFixed(1))

          // Primary sort by display score (higher first)
          if (bDisplayScore !== aDisplayScore) {
            return bDisplayScore - aDisplayScore
          }

          // Tie-breaker: if display scores are identical, use Brier score
          return (1 - b.avg_brier_score) - (1 - a.avg_brier_score)

        case 'brier_score':
          // Calculate display scores for Brier (rounded to 1 decimal place)
          const aBrierDisplay = parseFloat(((1 - a.avg_brier_score) * 100).toFixed(1))
          const bBrierDisplay = parseFloat(((1 - b.avg_brier_score) * 100).toFixed(1))

          // Primary sort by Brier display score (higher first)
          if (bBrierDisplay !== aBrierDisplay) {
            return bBrierDisplay - aBrierDisplay
          }

          // Tie-breaker: if display scores are identical, use PnL
          return b.final_cumulative_pnl - a.final_cumulative_pnl

        default:
          return 0
      }
    })
  }, [leaderboard, sortKey])


  const handleSort = (key: SortKey) => {
    setSortKey(key)
  }

  // Calculate min and max profit values for color scaling
  const profitRange = useMemo(() => {
    if (leaderboard.length === 0) return { min: 0, max: 0 }
    const profits = leaderboard.map(model => model.final_cumulative_pnl)
    return {
      min: Math.min(...profits),
      max: Math.max(...profits)
    }
  }, [leaderboard])

  const showMore = () => {
    setVisibleModels(prev => prev + 10)
  }

  return (
    <div>
      {/* Loading Spinner when initially loading */}
      {loading && leaderboard.length === 0 && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-2"></div>
            <div className="text-sm text-muted-foreground">Loading leaderboard...</div>
          </div>
        </div>
      )}

      <div className="bg-card rounded-xl border border-border/30 overflow-hidden max-w-4xl mx-auto">
        <div className="overflow-x-auto">
          <table className="w-full table-fixed">
            <thead className="bg-muted/30">
              <tr>
                <th className="text-center py-4 px-3 font-semibold w-12"></th>
                <th className="text-left py-4 px-4 font-semibold w-24">Model Name</th>
                <th className="hidden md:table-cell text-left py-4 px-4 font-semibold w-12"></th>
                <th className="text-center py-4 px-4 font-semibold w-24">
                  <div className="flex items-center justify-center space-x-1 w-full">
                    <button
                      onClick={() => handleSort('cumulative_profit')}
                      className="flex items-center space-x-1 hover:text-primary transition-colors whitespace-nowrap"
                    >
                      <ArrowDown className={`h-4 w-4 ${sortKey === 'cumulative_profit' ? 'text-primary' : 'opacity-40'}`} />
                      <span>Cumulative Profit</span>
                    </button>
                    <InfoTooltip content="This is the PnL (Profit and Loss), or cumulative profit from all trades made by the model" />
                  </div>
                </th>
                <th className="text-center py-4 px-4 font-semibold w-24">
                  <div className="flex items-center justify-center space-x-1 w-full">
                    <button
                      onClick={() => handleSort('brier_score')}
                      className="flex items-center space-x-1 hover:text-primary transition-colors whitespace-nowrap"
                      title="Brier Score - Higher values indicate better prediction accuracy (1 - original Brier score)"
                    >
                      <ArrowDown className={`h-4 w-4 ${sortKey === 'brier_score' ? 'text-primary' : 'opacity-40'}`} />
                      <span>Brier Score</span>
                    </button>
                    <InfoTooltip content="A measure of prediction accuracy. Lower values indicate better calibration - how well the model's confidence matches actual outcomes (0 = perfect, 1 = worst)" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {loading && leaderboard.length === 0 ? (
                Array.from({ length: 5 }).map((_, index) => (
                  <tr key={index} className="border-t border-border/20">
                    <td className="py-4 px-3 text-center">
                      <div className="h-4 bg-gray-200 rounded animate-pulse w-8 mx-auto"></div>
                    </td>
                    <td className="py-4 px-4">
                      <div className="h-4 bg-gray-200 rounded animate-pulse w-32"></div>
                    </td>
                    <td className="hidden md:table-cell py-4 px-4">
                      <div className="h-4 bg-gray-200 rounded animate-pulse w-24"></div>
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="h-4 bg-gray-200 rounded animate-pulse w-16 mx-auto"></div>
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="h-4 bg-gray-200 rounded animate-pulse w-16 mx-auto"></div>
                    </td>
                  </tr>
                ))
              ) : (
                sortedLeaderboard.slice(0, visibleModels).map((model, index) => (
                  <tr key={model.id} className="border-t border-border/20 hover:bg-muted/20 transition-colors">
                    <td className="py-4 px-3 text-center">
                      <span className={index <= 2 ? "text-2xl" : "text-md font-medium text-muted-foreground"}>
                        {index === 0 ? '🥇' :
                          index === 1 ? '🥈' :
                            index === 2 ? '🥉' :
                              `#${index + 1}`}
                      </span>
                    </td>
                    <td className="py-4 px-4">
                      <a
                        href={`/models?selected=${model.id}`}
                        className="font-medium hover:text-primary transition-colors whitespace-nowrap"
                      >
                        {model.model}
                      </a>
                    </td>
                    <td className="hidden md:table-cell py-4 px-4">
                      <CompanyDisplay modelName={model.model} />
                    </td>
                    <td className="py-4 px-4 text-center font-medium">
                      <a href={`/models?selected=${model.id}`} className="block">
                        <ProfitDisplay
                          value={model.final_cumulative_pnl}
                          minValue={profitRange.min}
                          maxValue={profitRange.max}
                          formatValue={(v) => `${(v * 100).toFixed(1)}%`}
                        />
                      </a>
                    </td>
                    <td className="py-4 px-4 text-center font-medium">
                      <a href={`/models?selected=${model.id}`} className="block">
                        {model.avg_brier_score ? `${((1 - model.avg_brier_score) * 100).toFixed(1)}%` : 'N/A'}
                      </a>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Show More Button */}
      {sortedLeaderboard.length > visibleModels && (
        <div className="text-center mt-6">
          <RedirectButton
            onClick={showMore}
            icon={<ChevronDown className="h-4 w-4" />}
          >
            Show more
          </RedirectButton>
        </div>
      )}
    </div>
  )
}