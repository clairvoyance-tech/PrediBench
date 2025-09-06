import { scaleLinear, scaleTime } from '@visx/scale'
import {
  AnimatedLineSeries,
  Axis,
  XYChart
} from '@visx/xychart'
import { extent } from 'd3-array'
import { format } from 'date-fns'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import styled from 'styled-components'

const tickLabelOffset = 10

interface DataPoint {
  x?: string | Date | null
  y?: number | null
  [key: string]: unknown
}

interface LineSeriesConfig {
  dataKey: string
  data: DataPoint[]
  stroke: string
  name?: string
}

interface VisxLineChartProps {
  height?: number
  margin?: { left: number; top: number; bottom: number; right: number }
  series: LineSeriesConfig[]
  xAccessor?: (d: DataPoint) => Date
  yAccessor?: (d: DataPoint) => number
  yDomain?: [number, number]
  formatTooltipX?: (value: Date) => string
  showGrid?: boolean
  numTicks?: number
}

const defaultAccessors = {
  // Be tolerant to null/undefined values to support discontinuities.
  xAccessor: (d: DataPoint) => {
    if (!d || d.x == null) return new Date(NaN)
    const val = d.x
    const date = val instanceof Date ? val : new Date(val as string)
    return isNaN(date.getTime()) ? new Date(NaN) : date
  },
  yAccessor: (d: DataPoint) => {
    if (!d || d.y == null) return NaN
    const num = Number(d.y)
    return Number.isFinite(num) ? num : NaN
  }
}

interface TooltipState {
  x: number
  y: number
  datum: DataPoint
  lineConfig: LineSeriesConfig
}

interface HoverState {
  xPosition: number
  tooltips: TooltipState[]
}

export function VisxLineChart({
  height = 270,
  margin = { left: 60, top: 35, bottom: 38, right: 27 },
  series,
  xAccessor = defaultAccessors.xAccessor,
  yAccessor = defaultAccessors.yAccessor,
  yDomain,
  formatTooltipX = (value: Date) => format(value, 'MMM d, yyyy'),
  showGrid = true,
  numTicks = 4
}: VisxLineChartProps) {
  // Ensure minimum of 4 ticks for better readability
  const effectiveNumTicks = Math.max(numTicks, 4)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoverState, setHoverState] = useState<HoverState | null>(null)
  const calculationQueueRef = useRef<{ x: number; timestamp: number }[]>([])
  const isProcessingRef = useRef<boolean>(false)

  const [containerWidth, setContainerWidth] = useState(800)

  // Safe wrappers to guard against bad data points provided by callers
  const safeXAccessor = useCallback(
    (d: DataPoint) => {
      try {
        const v = xAccessor(d)
        return v instanceof Date ? v : new Date(v as any)
      } catch {
        return new Date(NaN)
      }
    },
    [xAccessor]
  )

  const safeYAccessor = useCallback(
    (d: DataPoint) => {
      try {
        const v = yAccessor(d)
        const num = Number(v as any)
        return Number.isFinite(num) ? num : NaN
      } catch {
        return NaN
      }
    },
    [yAccessor]
  )

  // Update container width when component mounts/resizes
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        const width = rect.width || containerRef.current.offsetWidth || containerRef.current.clientWidth
        // Ensure minimum width to prevent zero-width chart
        const finalWidth = Math.max(width, 400)
        setContainerWidth(finalWidth)
      }
    }

    // Try multiple times to catch when DOM is ready
    updateWidth()
    setTimeout(updateWidth, 0)
    setTimeout(updateWidth, 100)

    // Also listen for resize events
    const resizeObserver = new ResizeObserver(updateWidth)
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => resizeObserver.disconnect()
  }, [])

  // Create scales for proper coordinate conversion
  const scales = useMemo(() => {
    const allData = series.flatMap(s => s.data).filter(d => d != null)
    if (allData.length === 0) return null

    const xExtent = extent(allData, safeXAccessor) as [Date, Date]
    let yExtent = yDomain || (extent(allData, safeYAccessor) as [number, number])

    // Simplified tick/domain calculation per 20% rule
    let actualTickCount = effectiveNumTicks
    let yTicks: number[] | undefined
    let yStep: number | undefined
    if (!yDomain) {
      const [dataMin, dataMax] = yExtent
      // Guard for degenerate ranges: handled via interval computation below

      // Nice intervals in ascending order
      const niceIntervals = [
        0.001, 0.002, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.25, 0.5,
        1, 2, 5, 10, 20, 50,
        100, 200, 500, 1000
      ]

      // Utility: compute tick domain and count for an interval
      const computeFor = (step: number) => {
        const maxAbove = Math.max(0, dataMax)
        const maxBelowAbs = Math.max(0, -Math.min(0, dataMin))
        const stepsAbove = Math.floor((maxAbove + 0.5 * step) / step)
        const stepsBelow = Math.floor((maxBelowAbs + 0.5 * step) / step)
        let lo = -stepsBelow * step
        let hi = stepsAbove * step
        if (hi === lo) hi = lo + step
        const count = Math.floor((hi - lo) / step) + 1
        return { lo, hi, count }
      }

      // Aim for 4-9 ticks; scan from SMALLEST interval and stop at the first that fits.
      const minTicks = 4
      const maxTicks = 9
      let chosen = null as null | { lo: number; hi: number; count: number; step: number }
      for (const ni of niceIntervals) {
        const r = computeFor(ni)
        if (r.count >= minTicks && r.count <= maxTicks) {
          chosen = { ...r, step: ni }
          break
        }
      }
      // Fallback to last interval if none matched
      if (!chosen) {
        const ni = niceIntervals[niceIntervals.length - 1]
        const r = computeFor(ni)
        chosen = { ...r, step: ni }
      }

      yExtent = [chosen.lo, chosen.hi]
      actualTickCount = chosen.count
      yStep = chosen.step
      // Generate explicit tick values to lock step
      if (Number.isFinite(yExtent[0]) && Number.isFinite(yExtent[1]) && Number.isFinite(yStep)) {
        const decimals = Math.max(0, (yStep.toString().split('.')[1]?.length || 0))
        const roundFix = (v: number) => Number(v.toFixed(decimals))
        const n = Math.max(2, Math.floor((yExtent[1] - yExtent[0]) / (yStep as number)) + 1)
        const t: number[] = []
        for (let i = 0; i < n; i++) t.push(roundFix(yExtent[0] + i * (yStep as number)))
        yTicks = t
      }
    } else {
      // yDomain provided: keep it as-is, but still compute ticks explicitly with a nice interval
      const [minD, maxD] = yDomain
      // If domain is invalid, subsequent code will fail naturally
      const niceIntervals = [
        0.001, 0.002, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.25, 0.5,
        1, 2, 5, 10, 20, 50,
        100, 200, 500, 1000
      ]
      const minTicks = 4
      const maxTicks = 9
      let chosenStep: number | null = null
      for (const ni of niceIntervals) {
        // ceil/floor to ensure ticks fall within domain
        const start = Math.ceil(minD / ni) * ni
        const end = Math.floor(maxD / ni) * ni
        const count = start <= end ? Math.floor((end - start) / ni) + 1 : 0
        if (count >= minTicks && count <= maxTicks) {
          chosenStep = ni
          break
        }
      }
      // No explicit fallback/errors
      // Build ticks within the provided domain bounds
      const step = chosenStep as number
      const decimals = Math.max(0, (step.toString().split('.')[1]?.length || 0))
      const roundFix = (v: number) => Number(v.toFixed(decimals))
      const start = Math.ceil(minD / step) * step
      const end = Math.floor(maxD / step) * step
      const n = Math.floor((end - start) / step) + 1
      yTicks = Array.from({ length: n }, (_, i) => roundFix(start + i * step))
      // Preserve the provided domain
      actualTickCount = yTicks.length
    }

    const xScale = scaleTime({
      domain: xExtent,
      range: [margin.left, containerWidth - margin.right]
    })

    const yScale = scaleLinear({
      domain: yExtent,
      range: [height - margin.bottom, margin.top]
    })

    return { xScale, yScale, yDomain: yExtent, actualTickCount, yTicks, yStep }
  }, [series, safeXAccessor, safeYAccessor, yDomain, margin, height, containerWidth, effectiveNumTicks])

  const processCalculationQueue = useCallback((targetX: number) => {
    if (!containerRef.current || !scales) return

    const hoveredTime = scales.xScale.invert(targetX)
    const newTooltips: TooltipState[] = []

    series.forEach((line) => {
      if (!line.data || line.data.length === 0) return

      // Consider only valid data points
      const validPoints = line.data.filter((p) => {
        const xd = safeXAccessor(p)
        const yd = safeYAccessor(p)
        return xd instanceof Date && !isNaN(xd.getTime()) && Number.isFinite(yd)
      })
      if (validPoints.length === 0) return

      // Skip this line if hovered x is outside this series' x-range (discontinuous support)
      const minX = validPoints.reduce((m, p) => Math.min(m, safeXAccessor(p).getTime()), Infinity)
      const maxX = validPoints.reduce((m, p) => Math.max(m, safeXAccessor(p).getTime()), -Infinity)
      const ht = hoveredTime.getTime()
      if (!(ht >= minX && ht <= maxX)) return

      // Find closest data point by time within this series
      let closestPoint = validPoints[0]
      let minDistance = Infinity
      for (const point of validPoints) {
        const t = safeXAccessor(point).getTime()
        const distance = Math.abs(t - ht)
        if (distance < minDistance) {
          minDistance = distance
          closestPoint = point
        }
      }

      // Use scales to get exact screen coordinates
      const xPos = scales.xScale(safeXAccessor(closestPoint))
      const yPos = scales.yScale(safeYAccessor(closestPoint))
      if (!Number.isFinite(xPos) || !Number.isFinite(yPos)) return

      newTooltips.push({
        x: xPos,
        y: yPos,
        datum: closestPoint,
        lineConfig: line
      })
    })

    // Filter out duplicate y=0 tooltips before setting state
    const filteredTooltips: TooltipState[] = []
    let hasSeenZero = false

    newTooltips.forEach(tooltip => {
      const yValue = safeYAccessor(tooltip.datum)
      // Check if the value would display as "0.00" when formatted with .toFixed(2)
      const displayValue = yValue.toFixed(2)
      const isDisplayZero = displayValue === '0.00'

      if (isDisplayZero) {
        if (!hasSeenZero) {
          filteredTooltips.push(tooltip)
          hasSeenZero = true
        }
        // Skip subsequent tooltips that display as 0.00
      } else {
        filteredTooltips.push(tooltip)
      }
    })

    // Use the x position from the first tooltip for the vertical line
    const alignedXPosition = filteredTooltips.length > 0 ? filteredTooltips[0].x : targetX

    setHoverState({
      xPosition: alignedXPosition,
      tooltips: filteredTooltips
    })
  }, [series, safeXAccessor, safeYAccessor, scales, containerRef])

  // Compute marker points: one circle per x-point, except the very first
  // point of each continuous segment. If clipTime is provided, only include
  // points with x <= clipTime so markers align with hover clipping.
  const getMarkerPoints = useCallback(
    (data: DataPoint[], clipTime?: Date) => {
      const points: DataPoint[] = []
      let inSegment = false
      for (const p of data) {
        const xd = safeXAccessor(p)
        const yd = safeYAccessor(p)
        const valid = xd instanceof Date && !isNaN(xd.getTime()) && Number.isFinite(yd)
        if (!valid) {
          inSegment = false
          continue
        }
        // entering a new segment: skip the first point
        if (!inSegment) {
          inSegment = true
          continue
        }
        if (clipTime) {
          if (xd.getTime() <= clipTime.getTime()) points.push(p)
        } else {
          points.push(p)
        }
      }
      return points
    },
    [safeXAccessor, safeYAccessor]
  )

  const handlePointerMove = useCallback((params: { event?: React.PointerEvent<Element> | React.FocusEvent<Element, Element>; svgPoint?: { x: number; y: number } }) => {
    if (!params.event || !containerRef.current || !scales) return

    const containerRect = containerRef.current.getBoundingClientRect()
    const mouseX = (params.event as React.PointerEvent<Element>).clientX - containerRect.left
    const now = Date.now()

    // Add to queue
    calculationQueueRef.current.push({ x: mouseX, timestamp: now })

    // Prune queue if it gets too large (keep only quantiles)
    const pruneThreshold = 5
    const quantiles = 4 // Can be adjusted: 4=quartiles, 10=deciles, etc.

    if (calculationQueueRef.current.length > pruneThreshold) {
      const queue = calculationQueueRef.current
      const prunedQueue: { x: number; timestamp: number }[] = []

      // Keep quantiles of the queue
      const step = Math.floor(queue.length / quantiles)
      for (let i = step - 1; i < queue.length; i += step) {
        prunedQueue.push(queue[i])
      }

      // Always keep the last item (most recent)
      if (prunedQueue[prunedQueue.length - 1] !== queue[queue.length - 1]) {
        prunedQueue.push(queue[queue.length - 1])
      }

      calculationQueueRef.current = prunedQueue
    }

    // Process queue if not already processing
    if (!isProcessingRef.current && calculationQueueRef.current.length > 0) {
      isProcessingRef.current = true

      const processNext = () => {
        if (calculationQueueRef.current.length === 0) {
          isProcessingRef.current = false
          return
        }

        // Take the most recent item from queue
        const item = calculationQueueRef.current.pop()!
        processCalculationQueue(item.x)

        // Continue processing if there are more items
        if (calculationQueueRef.current.length > 0) {
          setTimeout(processNext, 0) // Use setTimeout to avoid blocking
        } else {
          isProcessingRef.current = false
        }
      }

      processNext()
    }
  }, [processCalculationQueue])

  // Don't render chart until we have valid dimensions
  if (!scales || containerWidth < 100) {
    return (
      <ChartWrapper ref={containerRef}>
        <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
          Loading chart...
        </div>
      </ChartWrapper>
    )
  }

  return (
    <ChartWrapper
      ref={containerRef}
      onMouseLeave={() => {
        // Clear queue and state when mouse leaves
        calculationQueueRef.current = []
        isProcessingRef.current = false
        setHoverState(null)
      }}
    >
      <XYChart
        width={containerWidth}
        height={height}
        margin={margin}
        xScale={{ type: 'time' }}
        yScale={{ type: 'linear', domain: scales?.yDomain }}
        onPointerMove={handlePointerMove}
      >
        <defs>
          <clipPath id="reveal-clip">
            <rect
              x={margin.left}
              y={0}
              width="0"
              height={height}
              style={{
                animation: 'expandWidth 0.8s ease-out forwards'
              }}
            />
          </clipPath>

          {/* Dynamic hover clip paths for colored lines (horizontal-only clipping) */}
          {hoverState && series.map((_, index) => (
            <clipPath key={index} id={`hover-clip-${index}`}>
              <rect
                x={margin.left}
                y={0}
                width={Math.max(0, hoverState.xPosition - margin.left)}
                height={height}
              />
            </clipPath>
          ))}
        </defs>
        {/* Horizontal grid lines for each Y tick (locked to our computed ticks if available) */}
        {showGrid && (
          <g>
            {scales.yTicks!.map((t: number, i: number) => {
              const y = scales.yScale(t)
              return (
                <line
                  key={`grid-y-${i}`}
                  x1={margin.left}
                  x2={containerWidth - margin.right}
                  y1={y}
                  y2={y}
                  stroke="hsl(var(--border))"
                  strokeLinecap="round"
                  strokeWidth={1}
                  opacity={0.5}
                />
              )
            })}
          </g>
        )}

        <Axis
          hideAxisLine
          hideTicks
          orientation="bottom"
          tickLabelProps={() => ({ dy: tickLabelOffset })}
          numTicks={effectiveNumTicks}
        />
        <Axis
          hideAxisLine={false}
          hideTicks
          orientation="left"
          numTicks={scales.actualTickCount}
          tickValues={scales.yTicks}
          tickLabelProps={() => ({ dx: -10 })}
        />

        {series.map((line, index) => (
          <g key={line.dataKey}>
            {/* Gray background line - shows full line */}
            <AnimatedLineSeries
              stroke="hsl(var(--muted-foreground))"
              dataKey={`${line.dataKey}-gray`}
              data={line.data}
              xAccessor={safeXAccessor}
              yAccessor={safeYAccessor}
              style={{
                // Clip to chart plot area so nothing shows outside Y limits
                clipPath: 'url(#reveal-clip)'
              }}
            />

            {/* Colored line clipped to hover position */}
            <AnimatedLineSeries
              stroke={line.stroke}
              dataKey={`${line.dataKey}-colored`}
              data={line.data}
              xAccessor={safeXAccessor}
              yAccessor={safeYAccessor}
              style={{
                clipPath: hoverState ? `url(#hover-clip-${index})` : undefined
              }}
            />

            {/* Markers for colored series only: one per x-point except segment starts */}
            {(() => {
              const clipTime = hoverState ? scales.xScale.invert(hoverState.xPosition) : undefined
              const markerPoints = getMarkerPoints(line.data, clipTime)
              return (
                <g clipPath={hoverState ? `url(#hover-clip-${index})` : 'url(#reveal-clip)'}>
                  {markerPoints.map((pt, i) => {
                    const cx = scales.xScale(safeXAccessor(pt))
                    const cy = scales.yScale(safeYAccessor(pt))
                    if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null
                    return (
                      <circle
                        key={`${line.dataKey}-marker-${i}`}
                        cx={cx}
                        cy={cy}
                        r={3}
                        fill={line.stroke}
                        pointerEvents="none"
                      />
                    )
                  })}
                </g>
              )
            })()}
          </g>
        ))}
      </XYChart>

      {/* Hover state: single sliding container */}
      {hoverState && (() => {
        // Calculate anchoring once for the entire hover container
        const tooltipWidth = 150
        const chartWidth = containerWidth - margin.left - margin.right
        const anchorRight = hoverState.xPosition + tooltipWidth > margin.left + chartWidth

        return (
          <div
            style={{
              position: 'absolute',
              left: hoverState.xPosition,
              top: 0,
              pointerEvents: 'none',
              zIndex: 999,
              transform: anchorRight ? 'translateX(-100%)' : 'translateX(0%)'
            }}
          >
            {/* Vertical hover line */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: margin.top,
                width: '1px',
                backgroundColor: '#9ca3af',
                height: height - margin.top - margin.bottom
              }}
            />

            {/* Date label */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: margin.top - 20,
                transform: anchorRight ? 'translateX(-100%)' : 'translateX(0%)',
                color: '#9ca3af',
                fontSize: '11px',
                fontWeight: '500',
                whiteSpace: 'nowrap'
              }}
            >
              {hoverState.tooltips.length > 0 && formatTooltipX(safeXAccessor(hoverState.tooltips[0].datum))}
            </div>

            {/* Tooltips and hover points - positioned relative to container */}
            {(() => {
              // Sort from bottom to top (filtering is now done earlier)
              const sortedTooltips = [...hoverState.tooltips].sort((a, b) => b.y - a.y)

              // Position tooltips with overlap prevention
              const tooltipHeight = 24
              const gap = 2
              let lastBottom = height

              const repositionedTooltips = sortedTooltips.map(tooltip => {
                const originalTop = tooltip.y - tooltipHeight / 2
                let newTop = Math.min(originalTop, lastBottom - tooltipHeight - gap)
                newTop = Math.max(margin.top, newTop)
                lastBottom = newTop

                return {
                  ...tooltip,
                  adjustedY: newTop + tooltipHeight / 2,
                  // Convert absolute positions to relative positions within container
                  relativeX: tooltip.x - hoverState.xPosition
                }
              })

              return repositionedTooltips.map((tooltip, index) => (
                <div key={`tooltip-${tooltip.lineConfig.dataKey}-${index}`}>
                  {/* Hover point - positioned relative to container */}
                  <div
                    style={{
                      position: 'absolute',
                      left: tooltip.relativeX - 5,
                      top: tooltip.y - 5,
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      backgroundColor: tooltip.lineConfig.stroke,
                      border: '2px solid white',
                      zIndex: 1000
                    }}
                  />

                  {/* Tooltip - positioned relative to container */}
                  <div
                    style={{
                      position: 'absolute',
                      left: anchorRight ? tooltip.relativeX - 8 : tooltip.relativeX + 8,
                      top: tooltip.adjustedY,
                      transform: anchorRight ? 'translate(-100%, -50%)' : 'translateY(-50%)',
                      zIndex: 1001,
                      backgroundColor: tooltip.lineConfig.stroke,
                      color: 'white',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: '500',
                      whiteSpace: 'nowrap',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                  >
                    <strong>{safeYAccessor(tooltip.datum).toFixed(2)}</strong> - {(tooltip.lineConfig.name || tooltip.lineConfig.dataKey).substring(0, 20)}
                  </div>
                </div>
              ))
            })()}
          </div>
        )
      })()}
      {/* Debug overlay removed to keep code minimal */}
    </ChartWrapper>
  )
}

const ChartWrapper = styled.div`
  position: relative;
  max-width: 1000px;
  margin: 0 auto;
  
  text {
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
  }

  .visx-axis-tick {
    text {
      font-size: 12px;
      font-weight: 400;
      fill: hsl(var(--muted-foreground));
    }
  }
  
  @keyframes expandWidth {
    from {
      width: 0;
    }
    to {
      width: 100%;
    }
  }

  /* Responsive margins for mobile */
  @media (max-width: 768px) {
    margin-left: -1rem;
    margin-right: -1rem;
  }
`
