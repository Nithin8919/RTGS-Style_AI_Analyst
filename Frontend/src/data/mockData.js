// Mock data for government datasets
export const vehicleRegistrationData = [
  { month: 'Jan 2023', registrations: 3420, district: 'Hyderabad', vehicleType: 'Two Wheeler', fuelType: 'Petrol' },
  { month: 'Feb 2023', registrations: 3650, district: 'Hyderabad', vehicleType: 'Four Wheeler', fuelType: 'Diesel' },
  { month: 'Mar 2023', registrations: 4120, district: 'Warangal', vehicleType: 'Two Wheeler', fuelType: 'Electric' },
  { month: 'Apr 2023', registrations: 4350, district: 'Nizamabad', vehicleType: 'Four Wheeler', fuelType: 'CNG' },
  { month: 'May 2023', registrations: 4890, district: 'Hyderabad', vehicleType: 'Two Wheeler', fuelType: 'Petrol' },
  { month: 'Jun 2023', registrations: 5120, district: 'Warangal', vehicleType: 'Commercial', fuelType: 'Diesel' },
  { month: 'Jul 2023', registrations: 5450, district: 'Nizamabad', vehicleType: 'Two Wheeler', fuelType: 'Electric' },
  { month: 'Aug 2023', registrations: 5890, district: 'Hyderabad', vehicleType: 'Four Wheeler', fuelType: 'Petrol' },
  { month: 'Sep 2023', registrations: 6120, district: 'Warangal', vehicleType: 'Two Wheeler', fuelType: 'CNG' },
  { month: 'Oct 2023', registrations: 6450, district: 'Nizamabad', vehicleType: 'Commercial', fuelType: 'Diesel' },
  { month: 'Nov 2023', registrations: 6890, district: 'Hyderabad', vehicleType: 'Four Wheeler', fuelType: 'Electric' },
  { month: 'Dec 2023', registrations: 7250, district: 'Warangal', vehicleType: 'Two Wheeler', fuelType: 'Petrol' },
];

export const districtData = [
  { district: 'Hyderabad', total: 8245, perCapita: 12.4, rank: 1, growth: 15.2, population: 664000 },
  { district: 'Warangal', total: 3891, perCapita: 8.7, rank: 2, growth: 8.9, population: 447000 },
  { district: 'Nizamabad', total: 2156, perCapita: 6.3, rank: 3, growth: 12.1, population: 342000 },
  { district: 'Karimnagar', total: 1987, perCapita: 5.8, rank: 4, growth: 7.5, population: 342000 },
  { district: 'Khammam', total: 1654, perCapita: 4.9, rank: 5, growth: 9.2, population: 337000 },
  { district: 'Nalgonda', total: 1432, perCapita: 4.2, rank: 6, growth: 6.8, population: 341000 },
  { district: 'Adilabad', total: 1298, perCapita: 3.8, rank: 7, growth: 5.4, population: 341000 },
  { district: 'Mahbubnagar', total: 1156, perCapita: 3.4, rank: 8, growth: 8.1, population: 340000 },
  { district: 'Medak', total: 987, perCapita: 2.9, rank: 9, growth: 4.7, population: 340000 },
  { district: 'Ranga Reddy', total: 854, perCapita: 2.5, rank: 10, growth: 11.3, population: 342000 },
];

export const correlationMatrix = [
  { variable: 'Income', Income: 1.00, Population: 0.45, Urban: 0.78, Education: 0.82, Registrations: 0.67 },
  { variable: 'Population', Income: 0.45, Population: 1.00, Urban: 0.23, Education: 0.34, Registrations: 0.89 },
  { variable: 'Urban', Income: 0.78, Population: 0.23, Urban: 1.00, Education: 0.67, Registrations: 0.56 },
  { variable: 'Education', Income: 0.82, Population: 0.34, Urban: 0.67, Education: 1.00, Registrations: 0.45 },
  { variable: 'Registrations', Income: 0.67, Population: 0.89, Urban: 0.56, Education: 0.45, Registrations: 1.00 },
];

export const hypothesisTests = [
  { test: 'T-test', variables: 'Urban vs Rural Registrations', pValue: 0.003, significant: true, effectSize: 'Large (0.82)' },
  { test: 'Chi-square', variables: 'District vs Vehicle Type', pValue: 0.001, significant: true, effectSize: 'Medium (0.34)' },
  { test: 'ANOVA', variables: 'Income vs Registration Rate', pValue: 0.028, significant: true, effectSize: 'Small (0.18)' },
  { test: 'Pearson', variables: 'Population vs Registrations', pValue: 0.000, significant: true, effectSize: 'Large (0.89)' },
];

export const fuelTypeDistribution = [
  { name: 'Petrol', value: 65, color: '#FF9933' },
  { name: 'Diesel', value: 25, color: '#138808' },
  { name: 'Electric', value: 8, color: '#000080' },
  { name: 'CNG', value: 2, color: '#FF6B35' },
];

export const vehicleTypeDistribution = [
  { type: 'Two Wheeler', count: 28450 },
  { type: 'Four Wheeler', count: 12890 },
  { type: 'Commercial', count: 3235 },
  { type: 'Others', count: 655 },
];

export const sampleDatasets = [
  {
    id: 1,
    name: 'Telangana Vehicle Registrations 2023',
    domain: 'Transport',
    scope: 'Telangana 2023',
    rows: 45230,
    columns: 12,
    size: '2.4 MB',
    preview: [
      { District: 'Hyderabad', VehicleType: 'Two Wheeler', FuelType: 'Petrol', OwnerAge: 28, RegistrationDate: '2023-01-15' },
      { District: 'Warangal', VehicleType: 'Four Wheeler', FuelType: 'Diesel', OwnerAge: 35, RegistrationDate: '2023-01-16' },
      { District: 'Nizamabad', VehicleType: 'Two Wheeler', FuelType: 'Electric', OwnerAge: 24, RegistrationDate: '2023-01-17' },
    ]
  },
  {
    id: 2,
    name: 'Andhra Pradesh Hospital Capacity 2023',
    domain: 'Health',
    scope: 'Andhra Pradesh 2023',
    rows: 12450,
    columns: 8,
    size: '1.1 MB',
    preview: [
      { District: 'Visakhapatnam', HospitalType: 'Government', Beds: 450, Occupancy: 78, District: 'Urban' },
      { District: 'Vijayawada', HospitalType: 'Private', Beds: 320, Occupancy: 65, District: 'Urban' },
      { District: 'Guntur', HospitalType: 'Government', Beds: 280, Occupancy: 82, District: 'Rural' },
    ]
  },
  {
    id: 3,
    name: 'Karnataka School Enrollment 2023',
    domain: 'Education',
    scope: 'Karnataka 2023',
    rows: 28890,
    columns: 10,
    size: '1.8 MB',
    preview: [
      { District: 'Bangalore', SchoolType: 'Government', Enrollment: 1250, Dropout: 3.2, Medium: 'English' },
      { District: 'Mysore', SchoolType: 'Private', Enrollment: 890, Dropout: 2.1, Medium: 'Kannada' },
      { District: 'Hubli', SchoolType: 'Government', Enrollment: 567, Dropout: 4.5, Medium: 'Hindi' },
    ]
  }
];

export const runHistory = [
  {
    id: 'rtgs-transport-20250106-001',
    dataset: 'Telangana Vehicle Registrations 2023',
    domain: 'Transport',
    date: '2025-01-06',
    status: 'Completed',
    confidence: 8.7,
    duration: '2.4 min',
    favorite: true,
  },
  {
    id: 'rtgs-health-20250105-002',
    dataset: 'AP Hospital Capacity 2023',
    domain: 'Health', 
    date: '2025-01-05',
    status: 'Completed',
    confidence: 9.2,
    duration: '1.8 min',
    favorite: false,
  },
  {
    id: 'rtgs-education-20250104-003',
    dataset: 'Karnataka School Enrollment 2023',
    domain: 'Education',
    date: '2025-01-04',
    status: 'Failed',
    confidence: 0,
    duration: '0.3 min',
    favorite: false,
  },
];