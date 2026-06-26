import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:smart_park_fer_app/core/constants/colors.dart';
import 'package:smart_park_fer_app/core/services/auth_service.dart';

class SignupScreen extends StatefulWidget {
  const SignupScreen({super.key});

  @override
  State<SignupScreen> createState() => _SignupScreenState();
}

class _SignupScreenState extends State<SignupScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _birthdateController = TextEditingController();
  final TextEditingController _codeController = TextEditingController();

  bool _isConfirming = false;
  bool _isLoading = false;

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime(2000),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: const ColorScheme.light(
              primary: AppColors.forestGreen,
              onPrimary: Colors.white,
              onSurface: AppColors.charcoal,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _birthdateController.text = DateFormat('yyyy-MM-dd').format(picked);
      });
    }
  }

  bool _validateInputs() {
    final username = _usernameController.text.trim();
    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();
    final name = _nameController.text.trim();
    final birthdate = _birthdateController.text.trim();

    if (username.isEmpty || email.isEmpty || password.isEmpty || name.isEmpty || birthdate.isEmpty) {
      _showError("All fields are required");
      return false;
    }

    if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(email)) {
      _showError("Please enter a valid email address");
      return false;
    }

    if (password.length < 8) {
      _showError("Password must be at least 8 characters long");
      return false;
    }

    return true;
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red.shade400),
    );
  }

  Future<void> _handleSignup() async {
    if (!_validateInputs()) return;

    setState(() => _isLoading = true);
    final success = await AuthService().signUp(
      username: _usernameController.text.trim(),
      password: _passwordController.text.trim(),
      email: _emailController.text.trim(),
      name: _nameController.text.trim(),
      birthdate: _birthdateController.text.trim(),
    );
    setState(() => _isLoading = false);

    if (success) {
      setState(() => _isConfirming = true);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Registration successful! Please check your email for the confirmation code.")),
        );
      }
    } else {
      if (mounted) {
        _showError("Registration failed. Please try again.");
      }
    }
  }

  Future<void> _handleConfirm() async {
    if (_codeController.text.trim().isEmpty) {
      _showError("Please enter the confirmation code");
      return;
    }

    setState(() => _isLoading = true);
    final success = await AuthService().confirmSignUp(
      _usernameController.text.trim(),
      _codeController.text.trim(),
    );
    setState(() => _isLoading = false);

    if (success) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Account confirmed! You can now log in.")),
        );
        Navigator.of(context).pop(); // Back to login
      }
    } else {
      if (mounted) {
        _showError("Confirmation failed. Please check the code.");
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.creamBg,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.charcoal),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 32.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(
                Icons.person_add_outlined,
                size: 80,
                color: AppColors.forestGreen,
              ),
              const SizedBox(height: 24),
              Text(
                _isConfirming ? "CONFIRM ACCOUNT" : "CREATE ACCOUNT",
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 2,
                  color: AppColors.forestGreen,
                ),
              ),
              const SizedBox(height: 40),
              if (!_isConfirming) ...[
                _buildTextField(_usernameController, "Username", Icons.person_outline),
                const SizedBox(height: 16),
                _buildTextField(_emailController, "Email", Icons.email_outlined),
                const SizedBox(height: 16),
                _buildTextField(_passwordController, "Password", Icons.lock_outline, obscureText: true),
                const SizedBox(height: 16),
                _buildTextField(_nameController, "Full Name", Icons.badge_outlined),
                const SizedBox(height: 16),
                InkWell(
                  onTap: () => _selectDate(context),
                  child: IgnorePointer(
                    child: _buildTextField(_birthdateController, "Birthdate", Icons.calendar_today_outlined),
                  ),
                ),
                const SizedBox(height: 32),
                _buildButton("SIGN UP", _handleSignup),
              ] else ...[
                const Text(
                  "Enter the 6-digit code sent to your email",
                  textAlign: TextAlign.center,
                  style: TextStyle(color: AppColors.charcoal),
                ),
                const SizedBox(height: 24),
                _buildTextField(_codeController, "Confirmation Code", Icons.numbers),
                const SizedBox(height: 32),
                _buildButton("CONFIRM", _handleConfirm),
                TextButton(
                  onPressed: () => setState(() => _isConfirming = false),
                  child: const Text("Back to Sign Up", style: TextStyle(color: AppColors.forestGreen)),
                ),
              ],
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String label, IconData icon, {bool obscureText = false}) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon),
        border: const OutlineInputBorder(
          borderRadius: BorderRadius.all(Radius.circular(12)),
        ),
      ),
    );
  }

  Widget _buildButton(String text, VoidCallback onPressed) {
    return ElevatedButton(
      onPressed: _isLoading ? null : onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.forestGreen,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        elevation: 0,
      ),
      child: _isLoading
          ? const SizedBox(
              height: 20,
              width: 20,
              child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
            )
          : Text(
              text,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, letterSpacing: 2),
            ),
    );
  }
}
