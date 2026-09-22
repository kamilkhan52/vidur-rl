# Simulator Early Termination Fix

## Problem Description

At low request rates (high scale factors like 1.5), the Vidur simulator would exit prematurely with requests still pending in the scheduler. The symptoms were:

1. Event queue becomes empty
2. No more requests arriving (`get_next_request_arrival_time()` returns `None`)
3. Scheduler reports it's not empty (`is_empty()` returns `False`)
4. Simulation terminates with warning: "Simulation ended with pending requests in scheduler"

This happened because the while loop only checked:
- Event queue has events, OR
- More requests are arriving

But didn't account for the case where requests are waiting in the global scheduler's queue, waiting to be assigned to replicas, with no event to trigger that assignment.

## Root Cause

The simulator's main loop condition was:
```python
while not self._time_limit_reached and (
    self._event_queue
    or self._request_generator.get_next_request_arrival_time() is not None
):
```

At low request rates:
1. All requests arrive and get added to the global scheduler
2. Last RequestArrivalEvent fires, creating a GlobalScheduleEvent
3. That GlobalScheduleEvent might not assign all pending requests (replicas might be busy)
4. Replicas finish processing → BatchEndEvents → ReplicaScheduleEvents
5. Eventually, event queue drains while global scheduler still has unassigned requests
6. Loop exits because both conditions are false

## Solution

### Part 1: Extend Loop Condition
Added `or not self._scheduler.is_empty()` to the while loop condition:

```python
while not self._time_limit_reached and (
    self._event_queue
    or self._request_generator.get_next_request_arrival_time() is not None
    or not self._scheduler.is_empty()
):
```

This ensures the loop continues as long as there's work pending.

### Part 2: Handle Empty Event Queue with Pending Work
When the event queue is empty but the scheduler has work, manually inject a GlobalScheduleEvent:

```python
if not self._event_queue:
    # Only inject if no more requests arriving
    if self._request_generator.get_next_request_arrival_time() is None:
        # Prevent infinite loop
        if last_injected_global_schedule_time == self._time:
            logger.warning("Breaking to avoid infinite loop")
            break
        
        logger.debug("Injecting GlobalScheduleEvent")
        self._add_event(GlobalScheduleEvent(self._time))
        last_injected_global_schedule_time = self._time
        continue
```

### Part 3: Infinite Loop Prevention
Track the last time we injected a GlobalScheduleEvent. If we inject at time T and the queue is still empty after processing it (meaning no progress was made), break to avoid an infinite loop.

Reset the tracker when time advances:
```python
if last_injected_global_schedule_time is not None and event._time > last_injected_global_schedule_time:
    last_injected_global_schedule_time = None
```

## Testing

The fix was tested with a minimal simulation:
- Scheduler: LOP (batch-end)
- Model: Llama-2-7b
- Replicas: 16  
- Scale: 1.5 (low request rate)
- Trace: Synthetic (first 300 requests)

Expected behavior:
- Simulation completes successfully
- All requests processed
- No "pending requests" warnings
- Manual GlobalScheduleEvent injection occurs (visible in debug logs)
- No infinite loop breaks

## Files Modified

- `vidur/simulator.py`: Main fix in the `run()` method

## Future Considerations

The fix handles the immediate issue, but if the scheduler consistently cannot make progress (e.g., all replicas are at capacity and no batches are completing), the infinite loop break will trigger. This is a safety mechanism, but might indicate a deeper issue with the scheduler or configuration if it happens frequently.
