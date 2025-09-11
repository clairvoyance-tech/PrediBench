import { scaleLinear, scaleTime } from '@visx/scale'
import { Axis, XYChart } from '@visx/xychart'
import { extent } from 'd3-array'
import { format } from 'date-fns'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import styled from 'styled-components'
import type { ModelInvestmentDecision } from '../../api'
import { MarkerAnnotations } from './MarkerAnnotations'

const tickLabelOffset = 2

function SegmentedLineWithDots({
  data,
  xScale,
  yScale,
  xAccessor,
  yAccessor,
  stroke,
  clipPath,
  dotRadius = 3
}: {
  data: DataPoint[]
  xScale: (value: Date) => number
  yScale: (value: number) => number
  xAccessor: (d: DataPoint) => Date
  yAccessor: (d: DataPoint) => number
  stroke: string
  clipPath?: string
  dotRadius?: number
}) {
  // Build path commands for contiguous segments only
  const pathD: string[] = []
  let segmentOpen = false

  // Precompute points
  const points = data.map(d => ({
    x: xScale(xAccessor(d)),
    y: yScale(yAccessor(d)),
    raw: d
  }))

  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    const isValid = Number.isFinite(p.x) && Number.isFinite(p.y)
    if (!isValid) {
      segmentOpen = false
      continue
    }
    if (!segmentOpen) {
      pathD.push(`M ${p.x} ${p.y}`)
      segmentOpen = true
    } else {
      pathD.push(`L ${p.x} ${p.y}`)
    }
  }

  return (
    <g clipPath={clipPath} style={{ pointerEvents: 'none' }}>
      <path d={pathD.join(' ')} stroke={stroke} fill="none" />
      {points.map((p, idx) => (
        Number.isFinite(p.x) && Number.isFinite(p.y) ? (
          <circle key={`pt-${idx}`} cx={p.x} cy={p.y} r={dotRadius} fill={stroke} />
        ) : null
      ))}
    </g>
  )
}

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
  /** Optional: Show decision point markers with hover annotations */
  showDecisionMarkers?: boolean
  /** Optional: Investment decisions data for markers (required if showDecisionMarkers is true) */
  modelDecisions?: ModelInvestmentDecision[]
  /** Optional: Custom annotations indexed by date string (YYYY-MM-DD) */
  additionalAnnotations?: Record<string, {
    content: React.ReactNode
    /** Function to calculate the next annotation date for area highlighting */
    getNextDate?: () => string | null
  }>
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
  customAnnotation?: {
    date: Date
    content: React.ReactNode
    nextDate?: Date | null
  }
  clipEndPosition?: number  // For clipping colored line to end of annotation period
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
  numTicks = 4,
  showDecisionMarkers = false,
  modelDecisions = [],
  additionalAnnotations = {}
}: VisxLineChartProps) {
  // Ensure minimum of 4 ticks for better readability
  const effectiveNumTicks = Math.max(numTicks, 4)
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoverState, setHoverState] = useState<HoverState | null>(null)
  const calculationQueueRef = useRef<{ x: number; timestamp: number }[]>([])
  const isProcessingRef = useRef<boolean>(false)

  const [containerWidth, setContainerWidth] = useState(800)
  const isAnnotated = Object.keys(additionalAnnotations).length > 0
  const isMobile = containerWidth <= 768
  const chartHeight = isMobile && isAnnotated ? Math.round(height * 0.67) : height

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


  // Helper to find which annotation period a date falls into
  const findAnnotationPeriod = useCallback((date: Date): {
    startDate: Date
    endDate: Date | null
    annotation: typeof additionalAnnotations[string]
  } | null => {
    if (Object.keys(additionalAnnotations).length === 0) return null

    const dateTime = date.getTime()
    const annotationDates = Object.keys(additionalAnnotations).sort()

    // Find the annotation period this date falls into
    for (let i = 0; i < annotationDates.length; i++) {
      const startDate = new Date(annotationDates[i])
      const endDate = i < annotationDates.length - 1 ? new Date(annotationDates[i + 1]) : null

      // Check if date falls within this period
      const afterStart = dateTime >= startDate.getTime()
      const beforeEnd = !endDate || dateTime < endDate.getTime()

      if (afterStart && beforeEnd) {
        return {
          startDate,
          endDate,
          annotation: additionalAnnotations[annotationDates[i]]
        }
      }
    }

    return null
  }, [additionalAnnotations])

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
    console.log('[VisxLineChart] computing scales', { seriesCount: series.length })
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
      range: [chartHeight - margin.bottom, margin.top]
    })

    const debugTicks = (yTicks || []).slice(0, 10)
    console.log('[VisxLineChart] yDomain/ticks', { yExtent, actualTickCount, debugTicks })
    return { xScale, yScale, yDomain: yExtent, actualTickCount, yTicks, yStep }
  }, [series, safeXAccessor, safeYAccessor, yDomain, margin, chartHeight, containerWidth, effectiveNumTicks])

  const processCalculationQueue = useCallback((targetX: number) => {
    if (!containerRef.current || !scales) return

    const hoveredTime = scales.xScale.invert(targetX)

    // If additionalAnnotations is provided, completely disable standard tooltips
    if (Object.keys(additionalAnnotations).length > 0) {
      const period = findAnnotationPeriod(hoveredTime)

      if (period) {
        // Calculate the middle X position of the period for annotation display
        const startX = scales.xScale(period.startDate)
        const endX = period.endDate ? scales.xScale(period.endDate) : containerWidth - margin.right
        const middleX = (startX + endX) / 2

        const newState = {
          xPosition: middleX,
          tooltips: [],
          customAnnotation: {
            date: period.startDate,
            content: period.annotation.content,
            nextDate: period.endDate
          },
          clipEndPosition: endX
        }
        console.log('[VisxLineChart] setHoverState annotation', newState)
        setHoverState(newState)
        return
      }

      // No period found - no hover state when additionalAnnotations is provided
      setHoverState(null)
      return
    }

    // Standard tooltip logic (only when no additionalAnnotations)
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

    const newHover = {
      xPosition: alignedXPosition,
      tooltips: filteredTooltips
    }
    console.log('[VisxLineChart] setHoverState standard', newHover)
    setHoverState(newHover)
  }, [series, safeXAccessor, safeYAccessor, scales, containerRef, additionalAnnotations, findAnnotationPeriod, containerWidth, margin])

  // No separate marker computation. Circles are drawn in the same clipped groups
  // as their corresponding line segments to guarantee visual consistency.

  const handlePointerMove = useCallback((params: { event?: React.PointerEvent<Element> | React.FocusEvent<Element, Element>; svgPoint?: { x: number; y: number } }) => {
    if (!params.event || !containerRef.current || !scales) return

    const containerRect = containerRef.current.getBoundingClientRect()
    const mouseX = (params.event as React.PointerEvent<Element>).clientX - containerRect.left
    console.log('[VisxLineChart] pointerMove', { mouseX })
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
      {(() => { console.log('[VisxLineChart] render XYChart', { width: containerWidth, height: chartHeight, hoverStateExists: !!hoverState }); return null })()}
      <XYChart
        width={containerWidth}
        height={chartHeight}
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
              height={chartHeight}
              style={{
                animation: 'expandWidth 0.8s ease-out forwards'
              }}
            />
          </clipPath>

          {/* Dynamic hover clip paths for colored lines (horizontal-only clipping) */}
          {hoverState && series.map((_, index) => {
            // Use clipEndPosition if available (for annotation periods), otherwise use xPosition
            const clipX = hoverState.clipEndPosition ?? hoverState.xPosition
            const w = Math.max(0, clipX - margin.left)
            console.log('[VisxLineChart] hover clip rect', { index, clipX, width: w })
            return (
              <clipPath key={index} id={`hover-clip-${index}`}>
                <rect
                  x={margin.left}
                  y={0}
                  width={w}
                  height={chartHeight}
                />
              </clipPath>
            )
          })}
          {/* Hashed pattern for annotation period highlighting */}
          <pattern
            id="annotation-highlight"
            patternUnits="userSpaceOnUse"
            width="4"
            height="4"
            patternTransform="rotate(45)"
          >
            <rect
              width="4"
              height="4"
              fill="transparent"
            />
            <rect
              x="0"
              y="0"
              width="1"
              height="4"
              fill="hsl(var(--muted-foreground))"
              opacity="0.3"
            />
          </pattern>
        </defs>

        {/* Hashed area highlighting for annotation periods */}
        {hoverState?.customAnnotation && (
          <rect
            x={scales.xScale(hoverState.customAnnotation.date)}
            y={margin.top}
            width={hoverState.customAnnotation.nextDate
              ? scales.xScale(hoverState.customAnnotation.nextDate) - scales.xScale(hoverState.customAnnotation.date)
              : (containerWidth - margin.right) - scales.xScale(hoverState.customAnnotation.date)
            }
            height={chartHeight - margin.top - margin.bottom}
            fill="url(#annotation-highlight)"
            pointerEvents="none"
          />
        )}

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

        {/* Axes will be drawn after series to ensure visibility */}

        {/* Month separators and labels (below day ticks) */}
        {(() => {
          if (!scales) return null
          const [d0, d1] = scales.xScale.domain() as [Date, Date]
          const startDate = d0 instanceof Date ? d0 : new Date(d0)
          const endDate = d1 instanceof Date ? d1 : new Date(d1)

          // Compute month boundaries within domain (first day of each month)
          const boundaries: Date[] = []
          const firstBoundary = new Date(startDate.getFullYear(), startDate.getMonth() + 1, 1)
          for (let b = firstBoundary; b < endDate; b = new Date(b.getFullYear(), b.getMonth() + 1, 1)) {
            boundaries.push(new Date(b))
          }

          // Compute month spans for labeling
          const spans: { start: Date; end: Date; label: string }[] = []
          let mStart = new Date(startDate.getFullYear(), startDate.getMonth(), 1)
          while (mStart < endDate) {
            const mEnd = new Date(mStart.getFullYear(), mStart.getMonth() + 1, 1)
            spans.push({ start: new Date(mStart), end: new Date(mEnd), label: format(mStart, 'MMM') })
            mStart = mEnd
          }

          const plotLeft = margin.left
          const plotRight = containerWidth - margin.right
          // Position month bar/label below day labels but within SVG bounds
          const monthBarY = chartHeight - 17
          const monthLabelY = chartHeight - 3

          return (
            <g pointerEvents="none">
              {boundaries.map((bd, i) => {
                const x = scales.xScale(bd)
                if (!Number.isFinite(x)) return null
                return (
                  <line
                    key={`month-boundary-${i}`}
                    x1={x}
                    x2={x}
                    y1={margin.top}
                    y2={chartHeight - margin.bottom}
                    stroke="hsl(var(--border))"
                    strokeWidth={1}
                    opacity={0.5}
                  />
                )
              })}
              {spans.map((s, i) => {
                const sx = Math.max(plotLeft, scales.xScale(s.start))
                const ex = Math.min(plotRight, scales.xScale(s.end))
                if (!Number.isFinite(sx) || !Number.isFinite(ex)) return null
                const cx = (sx + ex) / 2
                return (
                  <g key={`month-label-${i}`}>
                    {/* Short bar above month label */}
                    <line
                      x1={cx - 10}
                      x2={cx + 10}
                      y1={monthBarY}
                      y2={monthBarY}
                      stroke="hsl(var(--muted-foreground))"
                      strokeWidth={2}
                      opacity={0.8}
                    />
                    <text
                      x={cx}
                      y={monthLabelY}
                      textAnchor="middle"
                      fontSize={12}
                      fill="hsl(var(--muted-foreground))"
                    >
                      {s.label}
                    </text>
                  </g>
                )
              })}
            </g>
          )
        })()}

        {series.map((line, index) => (
          <g key={line.dataKey}>
            {/* Gray background line with dots (full series, clipped to plot area reveal) */}
            <SegmentedLineWithDots
              data={line.data}
              xScale={scales.xScale}
              yScale={scales.yScale}
              xAccessor={safeXAccessor}
              yAccessor={safeYAccessor}
              stroke={"hsl(var(--muted-foreground))"}
              clipPath={'url(#reveal-clip)'}
            />

            {/* Colored line with dots, clipped to hover/annotation if present */}
            <SegmentedLineWithDots
              data={line.data}
              xScale={scales.xScale}
              yScale={scales.yScale}
              xAccessor={safeXAccessor}
              yAccessor={safeYAccessor}
              stroke={line.stroke}
              clipPath={hoverState ? `url(#hover-clip-${index})` : 'url(#reveal-clip)'}
            />
          </g>
        ))}

        {/* Axes drawn last to stay on top */}
        <Axis
          hideAxisLine
          hideTicks
          orientation="bottom"
          tickFormat={(d: any) => {
            try {
              const dt = d instanceof Date ? d : new Date(d)
              return format(dt, 'EEE dd')
            } catch {
              return ''
            }
          }}
          tickLabelProps={() => ({ dy: tickLabelOffset })}
          numTicks={effectiveNumTicks}
        />
        <Axis
          hideAxisLine={false}
          hideTicks
          orientation="left"
          numTicks={scales.actualTickCount}
          tickValues={scales.yTicks}
          tickFormat={(val: any) => {
            const v = typeof val === 'number' ? val : Number(val)
            if (!Number.isFinite(v)) return ''
            return `${Math.round(v * 100)}%`
          }}
          tickLabelProps={() => ({ dx: -10 })}
        />

        {/* Decision point markers with hover annotations */}
        {showDecisionMarkers && modelDecisions.length > 0 && (
          <MarkerAnnotations
            xScale={scales.xScale}
            yScale={scales.yScale}
            cumulativeData={series.length > 0 ? series[0].data.map(d => ({ x: d.x as string, y: d.y as number })) : []}
            modelDecisions={modelDecisions}
          />
        )}
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
            {/* Vertical hover lines - start and end of period */}
            {hoverState.customAnnotation ? (
              <>
                {/* Start line */}
                <div
                  style={{
                    position: 'absolute',
                    left: scales.xScale(hoverState.customAnnotation.date) - hoverState.xPosition,
                    top: margin.top,
                    width: '1px',
                    backgroundColor: '#9ca3af',
                    height: chartHeight - margin.top - margin.bottom
                  }}
                />
                {/* End line */}
                {hoverState.customAnnotation.nextDate && (
                  <div
                    style={{
                      position: 'absolute',
                      left: scales.xScale(hoverState.customAnnotation.nextDate) - hoverState.xPosition,
                      top: margin.top,
                      width: '1px',
                      backgroundColor: '#9ca3af',
                      height: chartHeight - margin.top - margin.bottom
                    }}
                  />
                )}
              </>
            ) : (
              /* Standard single vertical line */
              <div
                style={{
                  position: 'absolute',
                  left: 0,
                  top: margin.top,
                  width: '1px',
                  backgroundColor: '#9ca3af',
                  height: chartHeight - margin.top - margin.bottom
                }}
              />
            )}

            {/* Date label */}
            <div
              style={{
                position: 'absolute',
                left: hoverState.customAnnotation
                  ? scales.xScale(hoverState.customAnnotation.date) - hoverState.xPosition
                  : 0,
                top: margin.top - 20,
                transform: anchorRight ? 'translateX(-100%)' : 'translateX(0%)',
                color: '#9ca3af',
                fontSize: '11px',
                fontWeight: '500',
                whiteSpace: 'nowrap'
              }}
            >
              {hoverState.customAnnotation
                ? formatTooltipX(hoverState.customAnnotation.date)
                : hoverState.tooltips.length > 0 && formatTooltipX(safeXAccessor(hoverState.tooltips[0].datum))
              }
            </div>

            {/* Custom annotation or standard tooltips */}
            {hoverState.customAnnotation ? (
              /* Custom annotation display - positioned in middle of hashed area */
              <div
                style={{
                  position: 'absolute',
                  left: isMobile
                    ? (margin.left + ((containerWidth - margin.left - margin.right) / 2)) - hoverState.xPosition
                    : 0, // Already positioned at the middle X by processCalculationQueue
                  top: isMobile ? chartHeight + 8 : (margin.top + 40),
                  transform: 'translateX(-50%)', // Center horizontally on the anchor position
                  zIndex: 1001,
                  backgroundColor: 'hsl(var(--popover))',
                  color: 'hsl(var(--foreground))',
                  border: '1px solid hsl(var(--border))',
                  padding: isMobile ? '12px 14px' : '16px 20px',
                  borderRadius: '12px',
                  fontSize: '13px',
                  boxShadow: '0 8px 25px -3px rgba(0, 0, 0, 0.15)',
                  minWidth: isMobile ? `${Math.min(420, containerWidth - 24)}px` : '420px',
                  maxWidth: isMobile ? `${Math.max(420, containerWidth - 24)}px` : '500px',
                  whiteSpace: 'normal'
                }}
              >
                {hoverState.customAnnotation.content}
              </div>
            ) : (
              /* Standard tooltips */
              (() => {
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

                    {/* Standard tooltip */}
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
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '500',
                        whiteSpace: 'nowrap',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                      }}
                    >
                      <strong>{(safeYAccessor(tooltip.datum) * 100).toFixed(1)}%</strong> - {(tooltip.lineConfig.name || tooltip.lineConfig.dataKey).substring(0, 20)}
                    </div>
                  </div>
                ))
              })()
            )}
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
    /* Add space for annotation card below chart on mobile */
    padding-bottom: 170px;
  }
`
