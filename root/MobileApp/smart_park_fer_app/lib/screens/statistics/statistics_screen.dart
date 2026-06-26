import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:smart_park_fer_app/core/constants/colors.dart';
import 'package:smart_park_fer_app/models/node.dart';
import 'package:smart_park_fer_app/models/emotion_data.dart';
import 'package:smart_park_fer_app/screens/statistics/widgets/emotion_stats_widget.dart';
import 'package:smart_park_fer_app/core/services/data_service.dart';
import 'package:timezone/timezone.dart' as tz;

class StatisticsScreen extends StatefulWidget {
  const StatisticsScreen({super.key});

  @override
  State<StatisticsScreen> createState() => _StatisticsScreenState();
}

class _StatisticsScreenState extends State<StatisticsScreen> {
  final List<Node> _nodes = [
    Node(id: "1", name: "Tree1", emotionData: EmotionData()),
    Node(id: "2", name: "Tree2", emotionData: EmotionData()),
  ];

  Node? _selectedNode;
  DateTime? _selectedDate;
  DateTimeRange? _selectedDateRange;
  TimeOfDay? _startTime;
  TimeOfDay? _endTime;
  bool _showStats = false;
  bool _isLoading = false;
  EmotionData? _apiEmotionData;

  Future<void> _fetchData() async {
    if (_selectedNode == null || (_selectedDate == null && _selectedDateRange == null)) {
      return;
    }

    setState(() {
      _isLoading = true;
      _apiEmotionData = null;
    });

    try {
      int startTimestamp;
      int endTimestamp;

      final rome = tz.getLocation('Europe/Rome');

      if (_selectedDate != null) {
        final startHour = _startTime?.hour ?? 0;
        final startMinute = _startTime?.minute ?? 0;
        final endHour = _endTime?.hour ?? 23;
        final endMinute = _endTime?.minute ?? 59;

        // Start of day in Rome
        final startOfDay = tz.TZDateTime(
          rome,
          _selectedDate!.year,
          _selectedDate!.month,
          _selectedDate!.day,
          startHour,
          startMinute,
        );
        // End of day in Rome
        final endOfDay = tz.TZDateTime(
          rome,
          _selectedDate!.year,
          _selectedDate!.month,
          _selectedDate!.day,
          endHour,
          endMinute,
          59
        );
        
        startTimestamp = startOfDay.millisecondsSinceEpoch ~/ 1000;
        endTimestamp = endOfDay.millisecondsSinceEpoch ~/ 1000;
      } else {
        // Range start and end in Rome
        final startOfRange = tz.TZDateTime(
          rome,
          _selectedDateRange!.start.year,
          _selectedDateRange!.start.month,
          _selectedDateRange!.start.day,
        );
        final endOfRange = tz.TZDateTime(
          rome,
          _selectedDateRange!.end.year,
          _selectedDateRange!.end.month,
          _selectedDateRange!.end.day,
          23, 59, 59
        );

        startTimestamp = startOfRange.millisecondsSinceEpoch ~/ 1000;
        endTimestamp = endOfRange.millisecondsSinceEpoch ~/ 1000;
      }

      final results = await DataService().getEmotionData(
        start: startTimestamp,
        end: endTimestamp,
        nodename: _selectedNode!.name,
      );

      if (mounted) {
        setState(() {
          if (results.isNotEmpty) {
            // Aggregate all records into one (sum them up)
            int h = 0, s = 0, n = 0, sd = 0, f = 0, d = 0, c = 0, a = 0;
            for (var item in results) {
              h += item.happiness;
              s += item.surprise;
              n += item.neutral;
              sd += item.sadness;
              f += item.fear;
              d += item.disgust;
              c += item.contempt;
              a += item.anger;
            }
            _apiEmotionData = EmotionData(
              happiness: h, surprise: s, neutral: n, sadness: sd,
              fear: f, disgust: d, contempt: c, anger: a,
            );
          }
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error fetching data: $e")),
        );
      }
    }
  }

  void _onSelectNode(Node node) {
    setState(() {
      _selectedNode = node;
      _showStats = _selectedDate != null || _selectedDateRange != null;
    });
    if (_showStats) _fetchData();
  }

  Future<void> _pickDate() async {
    final picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: AppColors.forestGreen,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      if (!mounted) return;
      
      final choice = await showDialog<String>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text("Select Time Range"),
          content: const Text("Would you like to see data for the whole day or a specific time range?"),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, 'whole'),
              child: const Text("Whole Day", style: TextStyle(color: AppColors.forestGreen)),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, 'range'),
              child: const Text("Time Range", style: TextStyle(color: AppColors.forestGreen)),
            ),
          ],
        ),
      );

      if (choice == 'whole') {
        setState(() {
          _selectedDate = picked;
          _selectedDateRange = null;
          _startTime = null;
          _endTime = null;
          _showStats = true;
        });
        _fetchData();
      } else if (choice == 'range') {
        if (!mounted) return;
        final start = await showTimePicker(
          context: context,
          initialTime: const TimeOfDay(hour: 0, minute: 0),
          helpText: "Select Start Time",
        );
        if (start == null) return;

        if (!mounted) return;
        final end = await showTimePicker(
          context: context,
          initialTime: const TimeOfDay(hour: 23, minute: 59),
          helpText: "Select End Time",
        );
        if (end == null) return;

        if (start.hour > end.hour || (start.hour == end.hour && start.minute > end.minute)) {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Start time must be before end time")),
          );
          return;
        }

        setState(() {
          _selectedDate = picked;
          _selectedDateRange = null;
          _startTime = start;
          _endTime = end;
          _showStats = true;
        });
        _fetchData();
      }
    }
  }

  Future<void> _pickDateRange() async {
    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: AppColors.forestGreen,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _selectedDateRange = picked;
        _selectedDate = null;
        _startTime = null;
        _endTime = null;
        _showStats = true;
      });
      _fetchData();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.creamBg,
      
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            const Text(
              "Select a Node",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            SizedBox(
              height: 50,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _nodes.length,
                itemBuilder: (context, index) {
                  final node = _nodes[index];
                  final isSelected = _selectedNode?.id == node.id;
                  return Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: ChoiceChip(
                      label: Text(node.name),
                      selected: isSelected,
                      onSelected: (_) => _onSelectNode(node),
                      selectedColor: AppColors.forestGreen,
                      labelStyle: TextStyle(
                        color: isSelected ? Colors.white : AppColors.charcoal,
                      ),
                    ),
                  );
                },
              ),
            ),
            if (_selectedNode != null) ...[
              const SizedBox(height: 24),
              const Text(
                "Select Timeframe",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _pickDate,
                      icon: const Icon(Icons.today),
                      label: const Text("Specific Day"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: AppColors.forestGreen,
                        elevation: 0,
                        side: const BorderSide(color: AppColors.forestGreen),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _pickDateRange,
                      icon: const Icon(Icons.date_range),
                      label: const Text("Interval"),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: AppColors.forestGreen,
                        elevation: 0,
                        side: const BorderSide(color: AppColors.forestGreen),
                      ),
                    ),
                  ),
                ],
              ),
              if (_showStats) ...[
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  decoration: BoxDecoration(
                    color: AppColors.sageGreen,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.access_time, size: 16, color: AppColors.forestGreen),
                      const SizedBox(width: 8),
                      Text(
                        _selectedDate != null
                            ? "${DateFormat('MMM d, yyyy').format(_selectedDate!)}${_startTime != null ? " (${_startTime!.format(context)} - ${_endTime!.format(context)})" : " (Whole Day)"}"
                            : "${DateFormat('MMM d').format(_selectedDateRange!.start)} - ${DateFormat('MMM d, yyyy').format(_selectedDateRange!.end)}",
                        style: const TextStyle(
                          color: AppColors.forestGreen,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                if (_isLoading)
                  const Center(
                    child: Padding(
                      padding: EdgeInsets.symmetric(vertical: 40),
                      child: CircularProgressIndicator(color: AppColors.forestGreen),
                    ),
                  )
                else if (_apiEmotionData != null)
                  EmotionStatsWidget(
                    key: ValueKey(_apiEmotionData.hashCode),
                    title: _selectedNode!.name,
                    emotionData: _apiEmotionData!,
                  )
                else
                  const Center(
                    child: Padding(
                      padding: EdgeInsets.symmetric(vertical: 40),
                      child: Text("No data found for this timeframe", style: TextStyle(color: Colors.grey)),
                    ),
                  ),
              ],
            ],
            if (_selectedNode == null)
              const Center(
                child: Padding(
                  padding: EdgeInsets.only(top: 100),
                  child: Text(
                    "Please select a node to view statistics",
                    style: TextStyle(color: Colors.grey),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
