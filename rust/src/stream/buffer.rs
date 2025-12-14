//! High-performance ring buffer implementation

use crate::stream::MarketDataPoint;
use std::sync::Mutex;

/// High-performance ring buffer for market data
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");

        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);

        Self {
            buffer,
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }

    /// Push an item into the buffer
    pub fn push(&mut self, item: T) {
        self.buffer[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;

        if self.size < self.capacity {
            self.size += 1;
        } else {
            // Buffer is full, advance head
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Get the most recent n items
    pub fn get_recent(&self, n: usize) -> Vec<&T> {
        let mut result = Vec::with_capacity(n.min(self.size));

        if self.size == 0 {
            return result;
        }

        let start = if self.size < n {
            0
        } else {
            (self.tail + self.capacity - n) % self.capacity
        };

        let mut count = 0;
        let mut idx = start;
        while count < n.min(self.size) {
            if let Some(item) = &self.buffer[idx] {
                result.push(item);
                count += 1;
            }
            idx = (idx + 1) % self.capacity;
        }

        result
    }

    /// Get all items in the buffer
    pub fn get_all(&self) -> Vec<&T> {
        let mut result = Vec::with_capacity(self.size);

        if self.size == 0 {
            return result;
        }

        let mut idx = self.head;
        for _ in 0..self.size {
            if let Some(item) = &self.buffer[idx] {
                result.push(item);
            }
            idx = (idx + 1) % self.capacity;
        }

        result
    }

    /// Get the number of items in the buffer
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        for item in self.buffer.iter_mut() {
            *item = None;
        }
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Thread-safe ring buffer wrapper
pub struct ThreadSafeRingBuffer<T> {
    inner: Mutex<RingBuffer<T>>,
}

impl<T> ThreadSafeRingBuffer<T> {
    /// Create a new thread-safe ring buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(RingBuffer::new(capacity)),
        }
    }

    /// Push an item into the buffer
    pub fn push(&self, item: T) {
        if let Ok(mut buffer) = self.inner.lock() {
            buffer.push(item);
        }
    }

    /// Get recent items
    pub fn get_recent(&self, n: usize) -> Vec<T> {
        if let Ok(buffer) = self.inner.lock() {
            buffer.get_recent(n).into_iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get all items
    pub fn get_all(&self) -> Vec<T> {
        if let Ok(buffer) = self.inner.lock() {
            buffer.get_all().into_iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        if let Ok(buffer) = self.inner.lock() {
            buffer.len()
        } else {
            0
        }
    }
}

/// Sliding window buffer for time-based operations
pub struct SlidingWindowBuffer {
    buffer: RingBuffer<MarketDataPoint>,
    window_size_ms: u64,
}

impl SlidingWindowBuffer {
    /// Create a new sliding window buffer
    pub fn new(capacity: usize, window_size_ms: u64) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
            window_size_ms,
        }
    }

    /// Add a data point
    pub fn add(&mut self, data_point: MarketDataPoint) {
        self.buffer.push(data_point);
    }

    /// Get data points within the time window
    pub fn get_window(&self) -> Vec<&MarketDataPoint> {
        let current_time = chrono::Utc::now().timestamp_millis() as u64;
        let cutoff_time = current_time - self.window_size_ms;

        self.buffer.get_all()
            .into_iter()
            .filter(|data| data.timestamp as u64 >= cutoff_time)
            .collect()
    }

    /// Get data points for a specific symbol within the window
    pub fn get_window_for_symbol(&self, symbol: &str) -> Vec<&MarketDataPoint> {
        let current_time = chrono::Utc::now().timestamp_millis() as u64;
        let cutoff_time = current_time - self.window_size_ms;

        self.buffer.get_all()
            .into_iter()
            .filter(|data| {
                data.symbol == symbol && data.timestamp as u64 >= cutoff_time
            })
            .collect()
    }

    /// Get the count of items in the window
    pub fn window_size(&self) -> usize {
        self.get_window().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{MarketDataPoint, MarketDataType};

    #[test]
    fn test_ring_buffer_basic() {
        let mut buffer: RingBuffer<i32> = RingBuffer::new(3);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        // Add items
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.len(), 3);
        assert!(buffer.is_full());

        // Add another item (should overwrite oldest)
        buffer.push(4);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get_recent(2), vec![&3, &4]);
    }

    #[test]
    fn test_ring_buffer_get_recent() {
        let mut buffer: RingBuffer<String> = RingBuffer::new(5);

        for i in 1..=3 {
            buffer.push(format!("item_{}", i));
        }

        let recent = buffer.get_recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], "item_2");
        assert_eq!(recent[1], "item_3");

        let all = buffer.get_all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], "item_1");
        assert_eq!(all[1], "item_2");
        assert_eq!(all[2], "item_3");
    }

    #[test]
    fn test_sliding_window_buffer() {
        let mut buffer = SlidingWindowBuffer::new(100, 1000); // 1 second window

        // Add some old data
        let old_data = MarketDataPoint::new_trade("BTC/USDT".to_string(), 50000.0, 1.0);
        buffer.add(old_data);

        // All data should be in window initially
        assert_eq!(buffer.window_size(), 1);
    }

    #[test]
    fn test_thread_safe_ring_buffer() {
        let buffer = ThreadSafeRingBuffer::new(10);

        buffer.push(42);
        buffer.push(24);

        assert_eq!(buffer.len(), 2);

        let items = buffer.get_recent(1);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], 24);
    }
}