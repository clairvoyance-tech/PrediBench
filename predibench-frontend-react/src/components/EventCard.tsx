import { Clock, TrendingUp } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Event } from '../api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { formatVolume } from '../lib/utils'

interface EventCardProps {
  event: Event
}

export function EventCard({ event }: EventCardProps) {
  return (
    <Link key={event.id} to={`/events/${event.id}`}>
      <Card className="cursor-pointer hover:shadow-lg transition-all duration-200 hover:scale-[1.02]">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between mb-2">
            <CardTitle className="text-lg line-clamp-2 flex-1">{event.title}</CardTitle>
            <div className="flex items-center space-x-2 ml-2">
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                {event.end_datetime && new Date(event.end_datetime) > new Date() ? 'LIVE' : 'CLOSED'}
              </span>
            </div>
          </div>
          <CardDescription className="line-clamp-2 text-sm">
            {event.description || "No description available"}
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">{event.markets.length} market{event.markets.length !== 1 ? 's' : ''} in this event:</p>
              <div className="space-y-1">
                {event.markets?.slice(0, 2).map((market) => (
                  <div key={market.id} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground line-clamp-1 flex-1">
                      {market.question}
                    </span>
                    <div className="flex items-center space-x-2 ml-2">
                      <span className="font-medium text-xs">
                        {market.outcomes[0].price ? `${(market.outcomes[0].price * 100).toFixed(0)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>
                ))}
                {(event.markets.length) > 2 && (
                  <div className="text-xs text-muted-foreground text-center">
                    +{(event.markets.length) - 2} more market{(event.markets.length) - 2 !== 1 ? 's' : ''}
                  </div>
                )}
              </div>
            </div>
            
            <div className="flex items-center justify-between text-sm border-t pt-3">
              <div className="flex items-center space-x-4">
                <div className="flex items-center text-muted-foreground">
                  <span className="font-medium">
                    {formatVolume(event.volume)}
                  </span>
                </div>
                {event.liquidity && (
                  <div className="flex items-center text-muted-foreground whitespace-nowrap">
                    <TrendingUp className="h-4 w-4 mr-1" />
                    <span className="text-xs">
                      {formatVolume(event.liquidity)} liquidity
                    </span>
                  </div>
                )}
              </div>
              <div className="flex items-center text-muted-foreground ml-auto">
                <Clock className="h-4 w-4 mr-1" />
                <span className="text-xs">
                  {event.end_datetime
                    ? `Closes ${new Date(event.end_datetime).toLocaleDateString('en-US')}`
                    : 'No end date'
                  }
                </span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}