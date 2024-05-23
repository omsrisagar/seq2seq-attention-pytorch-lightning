use Statistics::ROC;
use Data::Dumper;
use Chart::Gnuplot;

# Chart object
my $chart = Chart::Gnuplot->new(
    output => "ROC_comparison.png",
#     imagesize => '3, 3',
#     xtics => {
#         font => "Sans, 30",
#     },
#     ytics => {
#         font => "Sans, 30",
#     },
    );

while (my $testanswer_data_filename = shift(@ARGV)) {
    
    # Run through the file once to find all the task names. Clunky, but arguably better than reading all the data into memory.
    open( my $testanswer_data_file, '<', $testanswer_data_filename ) or die "Can't open $testanswer_data_filename: $!";

    my @test_answers = ();

    while (my $line = <$testanswer_data_file>) {
        chomp($line);
        ($prediction,$true) = split(',', $line);
        push @test_answers, [$prediction, $true];
#        @test_answers = ([$prediction, $true], @test_answers);
    }

    print "Test answers: \n";
    print Dumper(@test_answers);

    my @curves = roc('increase', 0.95, @test_answers);

    print Dumper(@curves);
    print "End test answers for $testanswer_data_filename.\n";

    $roc_ref = $curves[1];

    $testanswer_data_filename =~ s/_/-/g;
    print "Filename: $testanswer_data_filename\n";
    
    # Data set object
    my $dataSet = Chart::Gnuplot::DataSet->new(
        points => $roc_ref,
        style => "lines",
        width => 5,
        title  => $testanswer_data_filename
        );

    push @dataSets, $dataSet;
}

# Plot the graph
$chart->plot2d(@dataSets);

