package qupath.lib.extension.monailabel.commands;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.gui.QuPathGUI;

/**
 * Command to run...
 * 
 * @author Sachidanand Alle
 *
 */
public class SubmitLabel implements Runnable {
	private final static Logger logger = LoggerFactory.getLogger(SubmitLabel.class);

	private QuPathGUI qupath;

	public SubmitLabel(QuPathGUI qupath) {
		this.qupath = qupath;
	}

	@Override
	public void run() {

	}

}